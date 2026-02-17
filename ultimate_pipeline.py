"""
Ultimate Suno Master Pipeline v6.3
===================================
9-Stage Professional Restoration for AI-Generated Audio.

Stage Order:
  1. Neural Clean      (DeepFilterNet 3)
  2. Spectral Shaper   (Magnitude — Ozone Stabilizer style)
  3. Phase Shaper      (Phase coherence: -1 linearize ← 0 → +1 analogize)
  4. Stereo Wider      (M/S enhancement + synthetic Side generation)
  5. Mono-Bass         (Linkwitz-Riley sub-bass → mono)
  6. Transient Punch   (Envelope-following gain mask)
  7. Spectre Restore   (Multi-band harmonic exciter)
  8. (Affinity Grouping — handled in notebook)
  9. Mastering Match   (Matchering reference)
"""

import os, re, shutil, torch, numpy as np
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F


def clean_name(filepath):
    """Strip Suno IDs and special chars from a filename."""
    base = os.path.basename(filepath).rsplit('.', 1)[0]
    clean = re.sub(r'^[\w\-]+?-', '', base)
    clean = re.sub(r'[\-\_\.]', ' ', clean)
    return clean.strip()


def build_track_name(index, filepath, ext='flac'):
    """Build a numbered output filename: '01 - Track Name.flac'"""
    num = str(index).zfill(2)
    name = clean_name(filepath)
    return num + ' - ' + name + '.' + ext


class UltimateSunoMaster:
    """7-Stage GPU-accelerated restoration pipeline for AI-generated audio."""

    def __init__(self, device='cuda', target_sr=48000, stages=None):
        self.device = device
        self.target_sr = target_sr
        self.stages = stages or {
            'neural_clean': True, 'spectral_shape': True, 'phase_shape': True,
            'stereo_widen': True, 'mono_bass': True, 'transient_punch': True,
            'spectre_restore': True,
        }
        self.dfn_available = False
        if self.stages.get('neural_clean'):
            try:
                from df.enhance import init_df, enhance as _enh
                self._dfn_model, self._df_state, _ = init_df()
                self._df_enhance = _enh
                self.dfn_available = True
                print('  ✅ DeepFilterNet 3 loaded')
            except Exception as e:
                print(f'  ⚠️ DFN3 unavailable: {e}')

    def _to_48k(self, wav, sr):
        if sr != self.target_sr:
            return T.Resample(sr, self.target_sr).to(self.device)(wav)
        return wav

    # ═════════════════════════════════════════════════════════
    #  Stage 1: Neural Cleaning (DeepFilterNet 3)
    # ═════════════════════════════════════════════════════════
    def neural_clean(self, wav):
        if not self.dfn_available:
            return wav
        try:
            return self._df_enhance(self._dfn_model, self._df_state, wav, atten_lim_db=6)
        except:
            return wav

    # ═════════════════════════════════════════════════════════
    #  Stage 2: Spectral Shaper — Ozone Stabilizer Mode
    #  (Magnitude-domain dynamic resonance control)
    # ═════════════════════════════════════════════════════════
    def spectral_shape(self, wav, amount=60, speed=90, sensitivity=30,
                       focus_low=200, focus_high=16000):
        sr = self.target_sr
        n_fft, hop = 4096, 1024
        win = torch.hann_window(n_fft).to(self.device)

        # Map 0-100 knobs to DSP values
        sens_db = 6.0 - (sensitivity / 100.0 * 5.0)
        max_cut = (amount / 100.0) * 8.0
        alpha = 0.05 + (speed / 100.0 * 0.45)

        channels = []
        for ch in range(wav.shape[0]):
            stft = torch.stft(wav[ch], n_fft=n_fft, hop_length=hop,
                              window=win, return_complex=True)
            mag, phase = stft.abs(), stft.angle()

            # Spectral envelope (moving average across freq bins)
            mag_t = mag.t().unsqueeze(1)
            k = torch.ones(1, 1, 31, device=self.device) / 31.0
            envelope = torch.nn.functional.conv1d(mag_t, k, padding=15).squeeze(1).t()

            # dB-domain peak detection
            mag_db = 20 * torch.log10(mag + 1e-8)
            env_db = 20 * torch.log10(envelope + 1e-8)
            excess = torch.clamp(mag_db - env_db - sens_db, min=0)

            # Soft-knee reduction
            reduction = torch.clamp(excess * 0.8, max=max_cut)
            gain = 10 ** (-reduction / 20)

            # Frequency focus mask
            freqs = torch.linspace(0, sr / 2, mag.shape[0]).to(self.device)
            mask = ((freqs >= focus_low) & (freqs <= focus_high)).float().unsqueeze(1)
            gain = gain * mask + (1.0 - mask)

            # Temporal smoothing (speed)
            for t in range(1, gain.shape[1]):
                gain[:, t] = alpha * gain[:, t] + (1.0 - alpha) * gain[:, t - 1]

            shaped = (mag * gain) * torch.exp(1j * phase)
            channels.append(torch.istft(shaped, n_fft=n_fft, hop_length=hop,
                                        window=win, length=wav.shape[-1]))
        return torch.stack(channels)

    # ═════════════════════════════════════════════════════════
    #  Stage 3: Spectral Phase Shaper
    #  control: -1.0 (linearize) ← 0.0 (bypass) → +1.0 (analogize)
    #  stereo_link: Apply correlated phase correction to both channels
    # ═════════════════════════════════════════════════════════
    def phase_shape(self, wav, control=-0.3, stereo_link=True):
        if abs(control) < 0.01:
            return wav

        sr = self.target_sr
        n_fft, hop = 4096, 1024
        win = torch.hann_window(n_fft).to(self.device)

        channels = []
        ref_delta = None  # For stereo link

        for ch in range(wav.shape[0]):
            stft = torch.stft(wav[ch], n_fft=n_fft, hop_length=hop,
                              window=win, return_complex=True)
            mag = stft.abs()
            phase = stft.angle()

            if control < 0:
                # ── LINEARIZE: reduce deviation from expected phase ──
                bin_idx = torch.arange(phase.shape[0], dtype=torch.float32,
                                       device=self.device)
                frame_idx = torch.arange(phase.shape[1], dtype=torch.float32,
                                         device=self.device)

                # Expected phase: progressive advance per bin per frame
                phase_advance = 2.0 * np.pi * bin_idx.unsqueeze(1) * hop / n_fft
                expected = phase[:, :1] + phase_advance * frame_idx.unsqueeze(0)

                # Wrapped deviation from expected
                delta = torch.atan2(torch.sin(phase - expected),
                                    torch.cos(phase - expected))

                factor = 1.0 + control  # -1→0.0 (full linear), 0→1.0 (original)

                if stereo_link and ch == 0:
                    ref_delta = delta.clone()

                if stereo_link and ch > 0 and ref_delta is not None:
                    # Apply same correction strength to maintain stereo image
                    # Only correct the "common" deviation, preserve the difference
                    common = ref_delta * factor
                    stereo_diff = delta - ref_delta
                    new_phase = expected + common + stereo_diff
                else:
                    new_phase = expected + delta * factor

            else:
                # ── ANALOGIZE: add hardware-inspired phase curve ──
                freqs = torch.linspace(0, sr / 2, phase.shape[0],
                                       device=self.device)

                analog = torch.zeros_like(freqs)
                # Transformer: low-freq phase lag
                analog += 0.15 * torch.exp(-((freqs - 80) / 60) ** 2)
                # Tube preamp: midrange rotation
                analog -= 0.08 * torch.exp(-((freqs - 3000) / 2000) ** 2)
                # Tape machine: HF shimmer
                analog += 0.12 * torch.sigmoid((freqs - 8000) / 2000)

                new_phase = phase + analog.unsqueeze(1) * control

            shaped = mag * torch.exp(1j * new_phase)
            channels.append(torch.istft(shaped, n_fft=n_fft, hop_length=hop,
                                        window=win, length=wav.shape[-1]))

        return torch.stack(channels)

    # ═════════════════════════════════════════════════════════
    #  Stage 4: Stereo Wider
    #  Frequency-dependent M/S widening with synthetic Side gen
    #  for tracks with little or no stereo content (common in Suno)
    # ═════════════════════════════════════════════════════════
    def stereo_widen(self, wav, width=0.3, generate_side=True):
        if wav.shape[0] < 2:
            wav = wav.expand(2, -1).clone()

        sr = self.target_sr
        n_fft, hop = 4096, 1024
        win = torch.hann_window(n_fft).to(self.device)

        L_stft = torch.stft(wav[0], n_fft=n_fft, hop_length=hop,
                            window=win, return_complex=True)
        R_stft = torch.stft(wav[1], n_fft=n_fft, hop_length=hop,
                            window=win, return_complex=True)

        # Mid/Side decomposition in frequency domain
        M = (L_stft + R_stft) / 2.0
        S = (L_stft - R_stft) / 2.0

        # Measure stereo content
        side_energy = S.abs().pow(2).mean()
        mid_energy = M.abs().pow(2).mean()
        stereo_ratio = (side_energy / (mid_energy + 1e-8)).item()

        freqs = torch.linspace(0, sr / 2, M.shape[0]).to(self.device)

        if generate_side and stereo_ratio < 0.05:
            # ── GENERATE SYNTHETIC SIDE (Haas-inspired spectral decorrelation) ──
            # Progressive phase offset: higher frequencies get more decorrelation
            phase_offset = torch.linspace(0.0, 0.8, M.shape[0]).to(self.device)
            # Protect sub-bass from decorrelation
            bass_protect = torch.sigmoid((freqs - 300) / 200)
            phase_offset = phase_offset * bass_protect

            # Synthetic side = fraction of mid with phase-shifted content
            synth_S = M.abs() * 0.18 * torch.exp(
                1j * (M.angle() + phase_offset.unsqueeze(1))
            )
            S = S + synth_S

        # ── Frequency-dependent width curve ──
        width_curve = torch.ones_like(freqs) * width
        # Bass: keep narrow for mono compatibility
        width_curve *= torch.sigmoid((freqs - 200) / 100)
        # Presence zone peak (2-8kHz): maximum perceived width
        width_curve *= 1.0 + 0.5 * torch.exp(-((freqs - 5000) / 3000) ** 2)
        # Ultra-high rolloff: avoid harshness
        width_curve *= 1.0 - 0.3 * torch.sigmoid((freqs - 14000) / 2000)

        # Apply width enhancement
        S_wide = S * (1.0 + width_curve.unsqueeze(1))

        # Reconstruct L/R from enhanced M/S
        L_new = M + S_wide
        R_new = M - S_wide

        L_out = torch.istft(L_new, n_fft=n_fft, hop_length=hop,
                            window=win, length=wav.shape[-1])
        R_out = torch.istft(R_new, n_fft=n_fft, hop_length=hop,
                            window=win, length=wav.shape[-1])

        result = torch.stack([L_out, R_out])
        peak = result.abs().max()
        return result * (0.98 / peak) if peak > 0.98 else result

    # ═════════════════════════════════════════════════════════
    #  Stage 5: Mono-Bass Phase Correction
    # ═════════════════════════════════════════════════════════
    def mono_bass(self, wav, cutoff_hz=150):
        sr = self.target_sr
        low = F.lowpass_biquad(F.lowpass_biquad(wav, sr, cutoff_hz), sr, cutoff_hz)
        high = wav - low
        if wav.shape[0] >= 2:
            low = low.mean(dim=0, keepdim=True).expand_as(low)
        return low + high

    # ═════════════════════════════════════════════════════════
    #  Stage 6: Transient Re-synthesis (Punch)
    # ═════════════════════════════════════════════════════════
    def transient_punch(self, wav, boost_db=4.0, release_ms=25):
        sr = self.target_sr
        mono = wav.mean(dim=0) if wav.shape[0] >= 2 else wav.squeeze(0)
        frame_len = int(sr * 0.005)
        hop = frame_len // 2
        if frame_len < 2:
            return wav

        padded = torch.nn.functional.pad(mono, (frame_len // 2, frame_len // 2))
        energy = padded.unfold(0, frame_len, hop).pow(2).mean(dim=-1).sqrt()

        flux = torch.clamp(torch.diff(energy, prepend=energy[:1]), min=0)
        if flux.max() < 1e-8:
            return wav
        flux_n = flux / (flux.max() + 1e-8)
        thr = flux_n.mean() + 1.5 * flux_n.std()
        mask = torch.clamp((flux_n - thr) / (1.0 - thr + 1e-8), 0, 1)

        gain = torch.nn.functional.interpolate(
            mask[None, None, :], size=wav.shape[-1], mode='linear', align_corners=False
        ).squeeze()

        rel = max(int(sr * release_ms / 1000), 4)
        k = torch.exp(-torch.arange(rel, device=self.device, dtype=torch.float32) / (rel / 4))
        k = (k / k.sum())[None, None, :]
        gain = torch.nn.functional.conv1d(
            gain[None, None, :], k, padding=rel // 2
        ).squeeze()[:wav.shape[-1]]

        boost = 10 ** (boost_db / 20)
        result = wav * (1.0 + gain.unsqueeze(0) * (boost - 1.0))
        peak = result.abs().max()
        return result * (0.98 / peak) if peak > 0.98 else result

    # ═════════════════════════════════════════════════════════
    #  Stage 7: Spectre High-End Restoration (Multi-Band Exciter)
    # ═════════════════════════════════════════════════════════
    def spectre_restore(self, wav):
        sr = self.target_sr
        stft = torch.stft(wav[0], n_fft=4096, hop_length=1024,
                          window=torch.hann_window(4096).to(self.device),
                          return_complex=True)
        mag_db = 20 * torch.log10(stft.abs().mean(dim=1) + 1e-8)
        freqs = torch.linspace(0, sr / 2, mag_db.shape[0]).to(self.device)
        valid = mag_db > (mag_db.max() - 55)
        cutoff = freqs[valid][-1].item() if valid.any() else 16000.0
        cutoff = max(12000.0, min(cutoff, 20000.0))
        if cutoff > 19500:
            return wav

        exc1 = F.highpass_biquad(
            torch.tanh(F.highpass_biquad(wav, sr, cutoff * 0.85) * 1.8),
            sr, cutoff * 0.9)
        exc2 = F.highpass_biquad(
            torch.tanh(F.highpass_biquad(wav, sr, cutoff) * 3.0),
            sr, cutoff)

        y = wav + (exc1 * 0.07) + (exc2 * 0.12)
        peak = y.abs().max()
        return y * (0.98 / peak) if peak > 0.98 else y

    # ═════════════════════════════════════════════════════════
    #  Full Pipeline Orchestrator
    # ═════════════════════════════════════════════════════════
    def process_track(self, input_path, output_path,
                      shaper_params=None, phase_control=-0.3,
                      stereo_width=0.3, punch_db=4.0, bass_hz=150):
        """GPU optimized processing with detailed stage telemetry."""
        current_stage = "Initialization"
        try:
            current_stage = "Loading"
            wav, sr = torchaudio.load(input_path)
            wav = self._to_48k(wav.to(self.device), sr)

            if self.stages.get('neural_clean'):
                current_stage = "Neural Clean (DFN3)"
                wav = self.neural_clean(wav)
            if self.stages.get('spectral_shape'):
                current_stage = "Spectral Shaper"
                p = shaper_params or {'amount': 60, 'speed': 90, 'sensitivity': 30}
                wav = self.spectral_shape(wav, **p)
            if self.stages.get('phase_shape'):
                current_stage = "Phase Shaper"
                wav = self.phase_shape(wav, control=phase_control)
            if self.stages.get('stereo_widen'):
                current_stage = "Stereo Wider"
                wav = self.stereo_widen(wav, width=stereo_width)
            if self.stages.get('mono_bass'):
                current_stage = "Mono-Bass"
                wav = self.mono_bass(wav, cutoff_hz=bass_hz)
            if self.stages.get('transient_punch'):
                current_stage = "Transient Punch"
                wav = self.transient_punch(wav, boost_db=punch_db)
            if self.stages.get('spectre_restore'):
                current_stage = "Spectre Restore"
                wav = self.spectre_restore(wav)

            current_stage = "Saving"
            torchaudio.save(output_path, wav.cpu(), self.target_sr,
                            encoding='PCM_S', bits_per_sample=16)
            return {"status": "ok", "stage": "Complete"}
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return {"status": "error", "stage": current_stage, "msg": str(e), "trace": error_details}


class MasteringEngine:
    """Stage 9: Reference-based mastering via Matchering."""

    def __init__(self, reference_path=None):
        self.ref = reference_path
        self.available = False
        if reference_path and os.path.exists(str(reference_path)):
            try:
                import matchering as _mg
                self._mg = _mg
                self.available = True
                print(f'  🎚️ Mastering ready | Ref: {os.path.basename(reference_path)}')
            except:
                print('  ⚠️ Matchering not installed')
        else:
            print('  ℹ️  Mastering disabled (no reference)')

    def master(self, in_path, out_path):
        if not self.available:
            if in_path != out_path:
                shutil.copy2(in_path, out_path)
            return False
        try:
            self._mg.process(target=in_path, reference=self.ref,
                             results=[self._mg.pcm16(out_path)])
            return True
        except Exception as e:
            print(f'  ⚠️ Mastering error: {e}')
            if in_path != out_path:
                shutil.copy2(in_path, out_path)
            return False
