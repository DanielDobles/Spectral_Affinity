"""
Ultimate Suno Master Pipeline v6.0
===================================
6-Stage Professional Restoration for AI-Generated Audio.

Stages:
  1. Neural Cleaning (DeepFilterNet 3) — artifact & noise removal
  2. Mono-Bass Phase Correction — solid low-end (< 150Hz → mono)
  3. Transient Re-synthesis (Punch) — restore dynamics & attack energy
  4. Spectre Restoration — multi-band harmonic exciter (48kHz)
  5. (Grouping handled by SpectralMasterEngine in notebook)
  6. Mastering Match (Matchering) — reference-based loudness & tonal balance
"""

import os, shutil, gc, torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F


# ═══════════════════════════════════════════════════════════════════
# Stage 1-4: UltimateSunoMaster
# ═══════════════════════════════════════════════════════════════════

class UltimateSunoMaster:
    """4-Stage GPU-accelerated restoration pipeline for AI-generated audio."""

    def __init__(self, device='cuda', target_sr=48000, stages=None):
        self.device = device
        self.target_sr = target_sr
        self.stages = stages or {
            'neural_clean': True,
            'mono_bass': True,
            'transient_punch': True,
            'spectre_restore': True,
        }

        # ── Init DeepFilterNet 3 ──
        self.dfn_available = False
        if self.stages.get('neural_clean'):
            try:
                from df.enhance import init_df, enhance as _df_enh
                self._dfn_model, self._df_state, _ = init_df()
                self._df_enhance = _df_enh
                self.dfn_available = True
                print("  ✅ DeepFilterNet 3 loaded")
            except Exception as e:
                print(f"  ⚠️ DeepFilterNet 3 unavailable: {e}")

    # ────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────
    def _to_48k(self, waveform, sr):
        if sr != self.target_sr:
            return T.Resample(sr, self.target_sr).to(self.device)(waveform)
        return waveform

    # ────────────────────────────────────────────────────────
    # Stage 1 — Neural Cleaning (DeepFilterNet 3)
    # ────────────────────────────────────────────────────────
    def neural_clean(self, waveform):
        """Conservative artifact removal. atten_lim_db=6 protects musical content."""
        if not self.dfn_available:
            return waveform
        try:
            enhanced = self._df_enhance(
                self._dfn_model, self._df_state, waveform, atten_lim_db=6
            )
            return enhanced
        except Exception as e:
            print(f"    ⚠️ DFN3 error: {e}")
            return waveform

    # ────────────────────────────────────────────────────────
    # Stage 2 — Mono-Bass Phase Correction
    # ────────────────────────────────────────────────────────
    def mono_bass(self, waveform, cutoff_hz=150):
        """Force sub-bass to mono via Linkwitz-Riley crossover."""
        sr = self.target_sr
        # Double-pass for steeper rolloff
        low = F.lowpass_biquad(waveform, sr, cutoff_hz)
        low = F.lowpass_biquad(low, sr, cutoff_hz)
        high = waveform - low
        # Sum low to mono if stereo
        if waveform.shape[0] >= 2:
            low_mono = low.mean(dim=0, keepdim=True).expand_as(low)
        else:
            low_mono = low
        return low_mono + high

    # ────────────────────────────────────────────────────────
    # Stage 3 — Transient Re-synthesis (Punch)
    # ────────────────────────────────────────────────────────
    def transient_punch(self, waveform, boost_db=4.0, release_ms=25):
        """Detect attack transients and restore their amplitude."""
        sr = self.target_sr
        mono = waveform.mean(dim=0) if waveform.shape[0] >= 2 else waveform.squeeze(0)

        # Short-term energy in 5ms frames
        frame_len = int(sr * 0.005)
        hop = frame_len // 2
        if frame_len < 2:
            return waveform

        padded = torch.nn.functional.pad(mono, (frame_len // 2, frame_len // 2))
        frames = padded.unfold(0, frame_len, hop)
        energy = frames.pow(2).mean(dim=-1).sqrt()

        # Positive spectral flux = onset
        flux = torch.diff(energy, prepend=energy[:1])
        flux = torch.clamp(flux, min=0)
        if flux.max() < 1e-8:
            return waveform

        # Normalize & threshold
        flux_n = flux / (flux.max() + 1e-8)
        thr = flux_n.mean() + 1.5 * flux_n.std()
        mask = torch.clamp((flux_n - thr) / (1.0 - thr + 1e-8), 0, 1)

        # Upsample gain mask to sample resolution
        gain = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=waveform.shape[-1], mode='linear', align_corners=False
        ).squeeze()

        # Release smoothing
        rel_samples = max(int(sr * release_ms / 1000), 4)
        kernel = torch.exp(
            -torch.arange(rel_samples, device=self.device, dtype=torch.float32) / (rel_samples / 4)
        )
        kernel = (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)
        gain = torch.nn.functional.conv1d(
            gain.unsqueeze(0).unsqueeze(0), kernel, padding=rel_samples // 2
        ).squeeze()[:waveform.shape[-1]]

        # Apply boost
        boost_lin = 10 ** (boost_db / 20)
        multiplier = 1.0 + gain * (boost_lin - 1.0)
        result = waveform * multiplier.unsqueeze(0)

        # Soft-clip
        peak = result.abs().max()
        if peak > 0.98:
            result = result * (0.98 / peak)
        return result

    # ────────────────────────────────────────────────────────
    # Stage 4 — Spectre High-End Restoration (Evolved)
    # ────────────────────────────────────────────────────────
    def spectre_restore(self, waveform):
        """Multi-band harmonic exciter for AI frequency gap recovery."""
        sr = self.target_sr

        # Detect cutoff via STFT magnitude analysis
        stft = torch.stft(
            waveform[0], n_fft=4096, hop_length=1024,
            window=torch.hann_window(4096).to(self.device),
            return_complex=True
        )
        mag_db = 20 * torch.log10(torch.abs(stft).mean(dim=1) + 1e-8)
        max_db = mag_db.max()
        freqs = torch.linspace(0, sr / 2, mag_db.shape[0]).to(self.device)
        mask = mag_db > (max_db - 55)
        cutoff = freqs[mask][-1].item() if mask.any() else 16000.0
        cutoff = max(12000.0, min(cutoff, 22000.0))

        if cutoff > 20000:
            return waveform  # Audio already has sufficient HF content

        # Band 1: Presence exciter (gentle)
        y_high = F.highpass_biquad(waveform, sr, cutoff * 0.85)
        y_exc1 = torch.tanh(y_high * 1.8)
        y_exc1 = F.highpass_biquad(y_exc1, sr, cutoff * 0.9)

        # Band 2: Air exciter (stronger)
        y_vhigh = F.highpass_biquad(waveform, sr, cutoff)
        y_exc2 = torch.tanh(y_vhigh * 3.0)
        y_exc2 = F.highpass_biquad(y_exc2, sr, cutoff)

        # Parallel mix
        y_final = waveform + (y_exc1 * 0.08) + (y_exc2 * 0.15)

        peak = y_final.abs().max()
        if peak > 0.98:
            y_final = y_final * (0.98 / peak)
        return y_final

    # ────────────────────────────────────────────────────────
    # Full Pipeline
    # ────────────────────────────────────────────────────────
    def process_track(self, input_path, output_path, verbose=True):
        """Run full 4-stage restoration on a single track → FLAC output."""
        try:
            waveform, sr = torchaudio.load(input_path)
            waveform = waveform.to(self.device)
            waveform = self._to_48k(waveform, sr)

            if self.stages.get('neural_clean') and self.dfn_available:
                if verbose: print("    🧹 Stage 1: Neural Cleaning...")
                waveform = self.neural_clean(waveform)

            if self.stages.get('mono_bass'):
                if verbose: print("    🔊 Stage 2: Mono-Bass...")
                waveform = self.mono_bass(waveform)

            if self.stages.get('transient_punch'):
                if verbose: print("    💥 Stage 3: Transient Punch...")
                waveform = self.transient_punch(waveform)

            if self.stages.get('spectre_restore'):
                if verbose: print("    ✨ Stage 4: Spectre Restoration...")
                waveform = self.spectre_restore(waveform)

            torchaudio.save(
                output_path, waveform.cpu(), self.target_sr,
                encoding="PCM_S", bits_per_sample=16
            )
            return True
        except Exception as e:
            print(f"    ⚠️ Restoration failed for {os.path.basename(input_path)}: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════
# Stage 6: MasteringEngine (Matchering)
# ═══════════════════════════════════════════════════════════════════

class MasteringEngine:
    """Reference-based mastering via Matchering (RMS, EQ, Peak, Stereo Width)."""

    def __init__(self, reference_path=None):
        self.reference = reference_path
        self.available = False
        if reference_path and os.path.exists(reference_path):
            try:
                import matchering as _mg
                self._mg = _mg
                self.available = True
                print(f"  🎚️ Mastering Engine ready | Ref: {os.path.basename(reference_path)}")
            except ImportError:
                print("  ⚠️ Matchering not installed")
        else:
            if reference_path:
                print(f"  ⚠️ Reference not found: {reference_path}")
            else:
                print("  ℹ️  Mastering Match disabled (no reference track)")

    def master(self, input_path, output_path):
        """Apply mastering matching. Falls back to copy on failure."""
        if not self.available:
            if input_path != output_path:
                shutil.copy2(input_path, output_path)
            return False
        try:
            self._mg.process(
                target=input_path,
                reference=self.reference,
                results=[self._mg.pcm16(output_path)]
            )
            return True
        except Exception as e:
            print(f"    ⚠️ Mastering error: {e}")
            if input_path != output_path:
                shutil.copy2(input_path, output_path)
            return False
