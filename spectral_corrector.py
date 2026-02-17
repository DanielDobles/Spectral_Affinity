import numpy as np
import librosa
import librosa.display
import soundfile as sf
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

class SpectralPhaseCorrector:
    """
    Surgical Spectral Phase Corrector
    Designed to detect and repair out-of-phase frequency components in stereo signals.
    """
    def __init__(self, n_fft=4096, hop_length=1024, sr=44100):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        
    def decompose(self, audio):
        """
        Step 1: Spectral Decomposition (STFT)
        Returns the complex spectrograms for L and R channels.
        """
        if audio.ndim == 1:
            raise ValueError("Audio must be stereo for phase correction.")
        
        # Split channels
        l_channel = audio[0, :]
        r_channel = audio[1, :]
        
        S_l = librosa.stft(l_channel, n_fft=self.n_fft, hop_length=self.hop_length)
        S_r = librosa.stft(r_channel, n_fft=self.n_fft, hop_length=self.hop_length)
        
        return S_l, S_r

    def analyze_phase(self, S_l, S_r):
        """
        Step 2: Vector Comparison & Coherence Analysis
        Calculates the phase difference and spectral coherence.
        """
        phase_l = np.angle(S_l)
        phase_r = np.angle(S_r)
        
        # Calculate phase difference (-pi to pi)
        phase_diff = phase_l - phase_r
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
        
        # Calculate Spectral Coherence (Correlation between L and R per bin)
        # Using a small temporal window to calculate local correlation
        mag_l = np.abs(S_l)
        mag_r = np.abs(S_r)
        
        # Cross-spectrum: S_l * conj(S_r)
        cross_spec = S_l * np.conj(S_r)
        
        # Coherence = Re(cross_spec) / (mag_l * mag_r)
        # This is essentially the cosine of the phase difference weighted by energy
        # 1.0 = perfect in-phase, -1.0 = perfect out-of-phase (cancelation)
        coherence = np.real(cross_spec) / (mag_l * mag_r + 1e-10)
        
        return phase_l, phase_r, phase_diff, coherence

    def calculate_mask(self, coherence, threshold_corr=0.0, freq_weighting=True):
        """
        Step 3: The Phase Wall
        Creates a correction mask based on coherence/correlation.
        Values near -1.0 are the 'Red Zone'.
        """
        # Threshold: if coherence < 0.0 (angle > 90 deg), we start correcting
        mask = (coherence < threshold_corr).astype(np.float32)
        
        # Frequency Weighting: Lower frequencies are more sensitive
        # We can increase the correction strength for low frequencies
        if freq_weighting:
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
            # Create a curve that is 1.0 at 0Hz and drops to 0.5 at 20kHz
            weight = np.exp(-freqs / 5000) + 0.5 
            mask = mask * weight[:, np.newaxis]
            
        # Clip mask to [0, 1]
        mask = np.clip(mask, 0, 1)
        
        return mask

    def apply_smoothing(self, mask, attack_ms=10, release_ms=50):
        """
        Step 5: The Glue (Musical Smoothing)
        Applies exponential smoothing (Attack/Release) to the mask
        to avoid rapid fluctuations (chatter).
        """
        # Convert MS to frames
        # frame_duration = hop_length / sr
        frame_time = self.hop_length / self.sr
        
        alpha_attack = np.exp(-frame_time / (attack_ms / 1000.0))
        alpha_release = np.exp(-frame_time / (release_ms / 1000.0))
        
        smoothed_mask = np.zeros_like(mask)
        # Initialize first frame
        smoothed_mask[:, 0] = mask[:, 0]
        
        # Apply smoothing across time (X axis / frames)
        for t in range(1, mask.shape[1]):
            target = mask[:, t]
            current = smoothed_mask[:, t-1]
            
            # If target > current, we are 'attacking' (increasing correction)
            # If target < current, we are 'releasing' (back to natural)
            is_attack = (target > current).astype(np.float32)
            alpha = is_attack * alpha_attack + (1 - is_attack) * alpha_release
            
            smoothed_mask[:, t] = alpha * current + (1 - alpha) * target
            
        # Optional: Light frequency smoothing
        from scipy.ndimage import gaussian_filter1d
        smoothed_mask = gaussian_filter1d(smoothed_mask, sigma=1, axis=0)
        
        return smoothed_mask

    def rotate_phase(self, S_r, phase_l, mask, smoothing_factor=0.1):
        """
        Step 4: The Elastic Rotator
        Surgically shifts phase of R channel bins that are in the 'red zone'.
        """
        mag_r = np.abs(S_r)
        phase_r = np.angle(S_r)
        
        # We want to pull phase_r towards phase_l if the mask is active
        # New phase: phase_r + mask * (delta_needed)
        # For simplicity, we can align it exactly with phase_l in bad zones
        # or just compress the difference.
        
        corrected_phase_r = phase_r.copy()
        
        # Soft-align: where mask is 1, steer phase_r towards phase_l
        # phase_l is the reference. 
        # Here we do a 'surgical' replacement:
        inner_limit = np.deg2rad(45) # We pull it into +/ 45 degrees of L
        
        # Linear interpolation of phase (careful with wrap-around)
        # Using complex vectors is safer for smoothing
        vec_l = np.exp(1j * phase_l)
        vec_r = np.exp(1j * phase_r)
        
        # Target vector is aligned with L
        # Corrected vector is a blend based on the mask
        corrected_vec_r = (1 - mask) * vec_r + mask * vec_l
        
        # Normalize magnitude
        corrected_vec_r = corrected_vec_r / (np.abs(corrected_vec_r) + 1e-10)
        
        # Reconstruct complex spectrum
        S_r_corrected = mag_r * corrected_vec_r
        
        return S_r_corrected

    def reconstruct(self, S_l, S_r_corrected):
        """
        Step 5: Reconstruction (ISTFT)
        """
        y_l = librosa.istft(S_l, hop_length=self.hop_length)
        y_r = librosa.istft(S_r_corrected, hop_length=self.hop_length)
        
        # Stack back to stereo
        # Ensure lengths match
        min_len = min(len(y_l), len(y_r))
        corrected_audio = np.vstack([y_l[:min_len], y_r[:min_len]])
        return corrected_audio

    def plot_phase_health(self, coherence, mask):
        """
        Visualizes the phase health (coherence) and the correction mask.
        """
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        librosa.display.specshow(coherence, x_axis='time', y_axis='hz', sr=self.sr, hop_length=self.hop_length, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label='Coherence (1=Safe, -1=Breaking)')
        plt.title('Spectral Coherence (Stereo Health)')
        
        plt.subplot(2, 1, 2)
        librosa.display.specshow(mask, x_axis='time', y_axis='hz', sr=self.sr, hop_length=self.hop_length, cmap='magma')
        plt.colorbar(label='Correction Intensity')
        plt.title('Correction Mask (Elastic Rotator Intensity)')
        
        plt.tight_layout()
        plt.show()

    def process(self, input_path, output_path, threshold_corr=0.0, plot=False):
        print(f"Loading: {input_path}")
        audio, sr = librosa.load(input_path, sr=self.sr, mono=False)
        
        print("Decomposing spectral components (STFT)...")
        S_l, S_r = self.decompose(audio)
        
        print("Analyzing phase vectors & spectral coherence...")
        phase_l, phase_r, phase_diff, coherence = self.analyze_phase(S_l, S_r)
        
        print(f"Applying threshold criteria (Coherence < {threshold_corr})...")
        mask = self.calculate_mask(coherence, threshold_corr)
        mask = self.apply_smoothing(mask)
        
        if plot:
            print("Generating phase health maps...")
            self.plot_phase_health(coherence, mask)
        
        print("Surgically rotating phase (Elastic Rotator)...")
        S_r_corrected = self.rotate_phase(S_r, phase_l, mask)
        
        print("Reconstructing stereo field (ISTFT)...")
        corrected_audio = self.reconstruct(S_l, S_r_corrected)
        
        print(f"Saving to: {output_path}")
        sf.write(output_path, corrected_audio.T, self.sr)
        print("Processing complete.")

if __name__ == "__main__":
    # Example usage (place holder)
    # corrector = SpectralPhaseCorrector()
    # corrector.process('input.wav', 'output_corrected.wav')
    pass
