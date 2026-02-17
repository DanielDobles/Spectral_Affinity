import os
import glob
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

def analyze_audio(file_path):
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None, mono=False)
        
        if y.ndim == 1:
            y_mono = y
            y_stereo = np.array([y, y])
        else:
            y_mono = librosa.to_mono(y)
            y_stereo = y

        # 1. Peak, RMS, Crest Factor
        peak = np.max(np.abs(y_mono))
        rms = np.sqrt(np.mean(y_mono**2))
        crest_factor = 20 * np.log10(peak / (rms + 1e-9)) if rms > 0 else 0

        # 2. DC Offset
        dc_offset = np.mean(y_mono)

        # 3. Phase Correlation & Stereo Width Ratio
        correlation = 1.0
        width_ratio = 0.0
        if y_stereo.shape[0] == 2:
            # Correlation
            corr_matrix = np.corrcoef(y_stereo[0], y_stereo[1])
            correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0,1]) else 1.0
            
            # Mid/Side Energy Ratio
            mid = (y_stereo[0] + y_stereo[1]) / 2
            side = (y_stereo[0] - y_stereo[1]) / 2
            mid_rms = np.sqrt(np.mean(mid**2))
            side_rms = np.sqrt(np.mean(side**2))
            width_ratio = side_rms / (mid_rms + 1e-9)

        # 4. Spectral Analysis
        S = np.abs(librosa.stft(y_mono))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Sub-bass energy (<40Hz) vs Low-mids (100-300Hz)
        sub_energy = np.mean(S[freqs < 40, :])
        low_mid_energy = np.mean(S[(freqs >= 100) & (freqs < 300), :])
        sub_bass_clutter = sub_energy / (low_mid_energy + 1e-9)
        
        # Sibilance / Harshness (5kHz - 10kHz)
        harshness = np.mean(S[(freqs >= 5000) & (freqs < 10000), :]) / (np.mean(S) + 1e-9)
        
        # HF Roll-off detection (Is there energy above 16kHz?)
        hf_energy = np.mean(S[freqs > 16000, :])
        total_energy = np.mean(S)
        hf_presence = hf_energy / (total_energy + 1e-9)

        # 5. Dynamic Range (Simple proxy)
        # Check for clipping (values >= 0.99)
        clipping_count = np.sum(np.abs(y_mono) >= 0.99)
        clipping_percent = (clipping_count / len(y_mono)) * 100

        return {
            "filename": os.path.basename(file_path),
            "peak_db": 20 * np.log10(peak + 1e-9),
            "rms_db": 20 * np.log10(rms + 1e-9),
            "crest_factor": crest_factor,
            "dc_offset": dc_offset,
            "phase_correlation": correlation,
            "stereo_width_ratio": width_ratio,
            "sub_bass_clutter": sub_bass_clutter,
            "harshness_ratio": harshness,
            "hf_presence": hf_presence,
            "clipping_percent": clipping_percent,
            "sample_rate": sr
        }
    except Exception as e:
        return {"filename": os.path.basename(file_path), "error": str(e)}

def main():
    dataset_path = r"c:\Users\armon\DEV_main\Spectral_Affinity\Slavic Data_Set"
    files = glob.glob(os.path.join(dataset_path, "*.*"))
    sample_files = [f for f in files if f.endswith(('.mp3', '.wav', '.flac'))]
    
    results = []
    print(f"Analyzing {len(sample_files)} files with advanced metrics...")
    for f in tqdm(sample_files):
        res = analyze_audio(f)
        results.append(res)
    
    # Aggregated Stats
    correlations = [r['phase_correlation'] for r in results if 'phase_correlation' in r]
    widths = [r['stereo_width_ratio'] for r in results if 'stereo_width_ratio' in r]
    clutter = [r['sub_bass_clutter'] for r in results if 'sub_bass_clutter' in r]
    harshness = [r['harshness_ratio'] for r in results if 'harshness_ratio' in r]
    hf_vals = [r['hf_presence'] for r in results if 'hf_presence' in r]
    crest_vals = [r['crest_factor'] for r in results if 'crest_factor' in r]
    clip_vals = [r['clipping_percent'] for r in results if 'clipping_percent' in r]

    print("\n" + "="*50)
    print("         ADVANCED AUDIO ANALYSIS SUMMARY")
    print("="*50)
    print(f"Avg Phase Correlation:     {np.mean(correlations):.3f} (Lower than 0.5 = Phase issues)")
    print(f"Avg Stereo Width Ratio:    {np.mean(widths):.3f} (S/M ratio)")
    print(f"Avg Crest Factor:          {np.mean(crest_vals):.2f} dB (Dynamics richness)")
    print(f"Avg Sub-Bass Clutter:      {np.mean(clutter):.3f} (Energy <40Hz / 100-300Hz)")
    print(f"Avg Harshness (5k-10k):    {np.mean(harshness):.3f}")
    print(f"Avg HF Presence (>16kHz):  {np.mean(hf_vals):.3f}")
    print(f"Avg Clipping Percent:      {np.mean(clip_vals):.4f}%")
    print("-" * 50)
    
    # Identify outliers
    print("\nOUTLIER DETECTION:")
    for r in results:
        if 'error' in r: continue
        issues = []
        if r['phase_correlation'] < 0.3: issues.append(f"Phase ({r['phase_correlation']:.2f})")
        if r['sub_bass_clutter'] > 1.0: issues.append(f"Sub-Rumble ({r['sub_bass_clutter']:.2f})")
        if r['harshness_ratio'] > 2.0: issues.append(f"Harshness ({r['harshness_ratio']:.2f})")
        if r['clipping_percent'] > 0.1: issues.append(f"Clipping ({r['clipping_percent']:.2f}%)")
        
        if issues:
            print(f"- {r['filename']}: {', '.join(issues)}")

if __name__ == "__main__":
    main()
