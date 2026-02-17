"""
Ultimate Suno Master Pipeline v6.2 — "Stabilizer Edition"
========================================================
7-Stage Professional Restoration with iZotope Ozone-style Spectral Shaping.
"""

import os, shutil, gc, torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F

class UltimateSunoMaster:
    def __init__(self, device='cuda', target_sr=48000, stages=None):
        self.device = device
        self.target_sr = target_sr
        self.stages = stages or {
            'neural_clean': True, 'spectral_shape': True, 'mono_bass': True,
            'transient_punch': True, 'spectre_restore': True,
        }
        self.dfn_available = False
        if self.stages.get('neural_clean'):
            try:
                from df.enhance import init_df, enhance as _df_enh
                self._dfn_model, self._df_state, _ = init_df()
                self._df_enhance = _df_enh
                self.dfn_available = True
            except: pass

    def _to_48k(self, waveform, sr):
        if sr != self.target_sr:
            return T.Resample(sr, self.target_sr).to(self.device)(waveform)
        return waveform

    def neural_clean(self, wav):
        if not self.dfn_available: return wav
        try: return self._df_enhance(self._dfn_model, self._df_state, wav, atten_lim_db=6)
        except: return wav

    # ── Stage 2: Spectral Shaper (Stabilizer Mode) ──
    def spectral_shape(self, wav, amount=50, speed=50, sensitivity=50, focus_low=200, focus_high=16000):
        """
        Ozone-style Spectral Shaper.
        - amount: Intensity factor (0-100)
        - speed: Temporal responsiveness (0-100 -> maps to alpha)
        - sensitivity: Detection threshold (0-100 -> maps to dB threshold)
        """
        sr = self.target_sr
        n_fft, hop = 4096, 1024
        win = torch.hann_window(n_fft).to(self.device)
        
        # Map parameters to DSP values
        sens_db = 6.0 - (sensitivity / 100.0 * 5.0) # 100 sens = 1dB threshold, 0 sens = 6dB
        max_cut = (amount / 100.0) * 8.0              # Max 8dB reduction
        alpha = 0.05 + (speed / 100.0 * 0.45)        # 0.05 (smooth) to 0.5 (fast)
        
        channels = []
        for ch in range(wav.shape[0]):
            stft = torch.stft(wav[ch], n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
            mag, phase = stft.abs(), stft.angle()
            
            # Local average for tonal balance
            mag_t = mag.t().unsqueeze(1)
            kernel = torch.ones(1, 1, 31, device=self.device) / 31
            envelope = torch.nn.functional.conv1d(mag_t, kernel, padding=15).squeeze(1).t()
            
            # Detect peaks
            mag_db, env_db = 20 * torch.log10(mag + 1e-8), 20 * torch.log10(envelope + 1e-8)
            excess = torch.clamp(mag_db - env_db - sens_db, min=0)
            
            # Reduction with soft-knee
            reduction_db = torch.clamp(excess * 0.8, max=max_cut)
            gain = 10 ** (-reduction_db / 20)
            
            # Frequency focus
            freqs = torch.linspace(0, sr/2, mag.shape[0]).to(self.device)
            mask = ((freqs >= focus_low) & (freqs <= focus_high)).float().unsqueeze(1)
            gain = gain * mask + (1.0 - mask)
            
            # Speed / Smoothing
            for t in range(1, gain.shape[1]):
                gain[:, t] = alpha * gain[:, t] + (1.0 - alpha) * gain[:, t-1]
                
            shaped = (mag * gain) * torch.exp(1j * phase)
            channels.append(torch.istft(shaped, n_fft=n_fft, hop_length=hop, window=win, length=wav.shape[-1]))
            
        return torch.stack(channels)

    def mono_bass(self, wav, cutoff_hz=150):
        sr = self.target_sr
        low = F.lowpass_biquad(F.lowpass_biquad(wav, sr, cutoff_hz), sr, cutoff_hz)
        high = wav - low
        if wav.shape[0] >= 2: low = low.mean(dim=0, keepdim=True).expand_as(low)
        return low + high

    def transient_punch(self, wav, boost_db=4.0):
        sr = self.target_sr
        mono = wav.mean(dim=0) if wav.shape[0] >= 2 else wav.squeeze(0)
        f, h = int(sr * 0.005), int(sr * 0.005) // 2
        p = torch.nn.functional.pad(mono, (f//2, f//2))
        e = p.unfold(0, f, h).pow(2).mean(dim=-1).sqrt()
        flux = torch.clamp(torch.diff(e, prepend=e[:1]), min=0)
        if flux.max() < 1e-8: return wav
        mask = torch.clamp((flux/(flux.max()+1e-8) - 0.2)/0.8, 0, 1)
        gain = torch.nn.functional.interpolate(mask[None, None, :], size=wav.shape[-1], mode='linear').squeeze()
        boost = 10 ** (boost_db / 20)
        res = wav * (1.0 + gain.unsqueeze(0) * (boost - 1.0))
        peak = res.abs().max()
        return res * (0.98/peak) if peak > 0.98 else res

    def spectre_restore(self, wav):
        sr = self.target_sr
        stft = torch.stft(wav[0], n_fft=4096, hop_length=1024, window=torch.hann_window(4096).to(self.device), return_complex=True)
        mag = 20 * torch.log10(stft.abs().mean(dim=1) + 1e-8)
        freqs = torch.linspace(0, sr/2, mag.shape[0]).to(self.device)
        mask = mag > (mag.max() - 55)
        cutoff = freqs[mask][-1].item() if mask.any() else 16000.0
        cutoff = max(12000.0, min(cutoff, 20000.0))
        if cutoff > 19500: return wav
        exc1 = F.highpass_biquad(torch.tanh(F.highpass_biquad(wav, sr, cutoff*0.8) * 1.8), sr, cutoff*0.9)
        exc2 = F.highpass_biquad(torch.tanh(F.highpass_biquad(wav, sr, cutoff) * 3.0), sr, cutoff)
        y = wav + (exc1 * 0.07) + (exc2 * 0.12)
        peak = y.abs().max()
        return y * (0.98/peak) if peak > 0.98 else y

    def process_track(self, input_path, output_path, shaper_params=None):
        try:
            wav, sr = torchaudio.load(input_path)
            wav = self._to_48k(wav.to(self.device), sr)
            if self.stages.get('neural_clean'):   wav = self.neural_clean(wav)
            if self.stages.get('spectral_shape'): 
                params = shaper_params or {'amount': 50, 'speed': 50, 'sensitivity': 50}
                wav = self.spectral_shape(wav, **params)
            if self.stages.get('mono_bass'):      wav = self.mono_bass(wav)
            if self.stages.get('transient_punch'): wav = self.transient_punch(wav)
            if self.stages.get('spectre_restore'): wav = self.spectre_restore(wav)
            torchaudio.save(output_path, wav.cpu(), 48000, encoding="PCM_S", bits_per_sample=16)
            return True
        except Exception as e:
            print(f"  ⚠️ Error: {e}"); return False

class MasteringEngine:
    def __init__(self, reference_path=None):
        self.ref = reference_path
        self.available = reference_path and os.path.exists(reference_path)
        if self.available:
            try: import matchering as mg; self._mg = mg; print(f"  🎚️ Master Match ready")
            except: self.available = False
    def master(self, in_p, out_p):
        if not self.available: shutil.copy2(in_p, out_p); return False
        try: self._mg.process(target=in_p, reference=self.ref, results=[self._mg.pcm16(out_p)]); return True
        except: shutil.copy2(in_p, out_p); return False
