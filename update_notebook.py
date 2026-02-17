import json
import os

nb_path = r'c:\Users\armon\DEV_main\Spectral_Affinity\Spectral_Affinity_Master.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Header
new_header = [
    "# 🌌 Spectral Affinity Master v6.4\n",
    "### 9-Stage Ultimate SunoMaster Pipeline (Audiophile Edition)\n",
    "\n",
    "| # | Stage | Tech | Effect |\n",
    "|---|-------|------|--------|\n",
    "| 1 | 🧹 Soft Gate | Spectral Expander | Gentle noise floor reduction (No DFN) |\n",
    "| 2 | 🎛️ Spectral Shaper | STFT Stabilizer | Resonance / harshness control |\n",
    "| 3 | 🌀 Phase Correct | Coherence Rotator | Fixes phase cancellation (Non-destructive) |\n",
    "| 4 | 🔀 Stereo Wider | Safe M/S | Width without phase issues |\n",
    "| 5 | 🔊 Mono-Bass | Linkwitz-Riley | Sub-bass phase → mono |\n",
    "| 6 | 💥 Transient Punch | Envelope Mask | Restore attack dynamics |\n",
    "| 7 | ✨ Spectre Restore | Harmonic Exciter | 48kHz high-end recovery |\n",
    "| 8 | 🏝️ Affinity Grouping | MERT + K-Means | Neural semantic clustering |\n",
    "| 9 | 🎚️ Mastering Match | Matchering | Reference loudness & tone |"
]

# 2. Update Restoration Class
new_restoration_code = r"""class UltimateSunoMaster:
    def __init__(self, device='cuda', target_sr=48000, stages=None):
        self.device, self.target_sr = device, target_sr
        self.stages = stages or {'neural_clean':True,'spectral_shape':True,'phase_shape':True,'stereo_widen':True,'mono_bass':True,'transient_punch':True,'spectre_restore':True}
        # DFN Removed per user request

    def _to_48k(self, wav, sr):
        try:
            return T.Resample(sr, self.target_sr).to(self.device)(wav) if sr != self.target_sr else wav
        except: return wav

    # ── Stage 1: Soft Spectral Gate (Replaces Neural Clean) ──
    def neural_clean(self, wav, threshold_db=-60, reduction_db=12):
        if not self.stages.get('neural_clean'): return wav
        sr, n, h = self.target_sr, 4096, 1024
        win = torch.hann_window(n).to(self.device)
        chs = []
        for ch in range(wav.shape[0]):
            stft = torch.stft(wav[ch], n_fft=n, hop_length=h, window=win, return_complex=True)
            mag, phase = stft.abs(), stft.angle()
            mag_db = 20 * torch.log10(mag + 1e-8)
            mask = torch.sigmoid((mag_db - threshold_db) * 2.0)
            gain = mask + 0.25 * (1 - mask)
            chs.append(torch.istft(mag * gain * torch.exp(1j*phase), n_fft=n, hop_length=h, window=win, length=wav.shape[-1]))
        return torch.stack(chs)

    # ── Stage 2: Spectral Shaper (Stabilizer) ──
    def spectral_shape(self, wav, amount=60, speed=90, sensitivity=30, focus_low=200, focus_high=16000):
        if not self.stages.get('spectral_shape'): return wav
        sr, n, h = self.target_sr, 4096, 1024
        win = torch.hann_window(n).to(self.device)
        sens_db = 6.0 - (sensitivity/100.0*5.0); max_cut = (amount/100.0)*8.0; alpha = 0.05+(speed/100.0*0.45)
        chs = []
        for ch in range(wav.shape[0]):
            stft = torch.stft(wav[ch], n_fft=n, hop_length=h, window=win, return_complex=True)
            mag, phase = stft.abs(), stft.angle()
            env = torch.nn.functional.conv1d(mag.t().unsqueeze(1), torch.ones(1,1,31,device=self.device)/31, padding=15).squeeze(1).t()
            excess = torch.clamp(20*torch.log10(mag+1e-8) - 20*torch.log10(env+1e-8) - sens_db, min=0)
            gain = 10**(-torch.clamp(excess*0.8, max=max_cut)/20)
            freqs = torch.linspace(0, sr/2, mag.shape[0]).to(self.device)
            mask = ((freqs>=focus_low)&(freqs<=focus_high)).float().unsqueeze(1)
            gain = gain*mask + (1.0-mask)
            for t in range(1, gain.shape[1]): gain[:,t] = alpha*gain[:,t] + (1-alpha)*gain[:,t-1]
            chs.append(torch.istft((mag*gain)*torch.exp(1j*phase), n_fft=n, hop_length=h, window=win, length=wav.shape[-1]))
        return torch.stack(chs)

    # ── Stage 3: Phase Corrector (Replaces Destructive Phase Shaper) ──
    def phase_shape(self, wav, threshold_corr=0.0):
        if not self.stages.get('phase_shape') or wav.shape[0] < 2: return wav
        sr, n, h = self.target_sr, 4096, 1024
        win = torch.hann_window(n).to(self.device)
        S_l = torch.stft(wav[0], n_fft=n, hop_length=h, window=win, return_complex=True)
        S_r = torch.stft(wav[1], n_fft=n, hop_length=h, window=win, return_complex=True)
        mag_l, mag_r = S_l.abs(), S_r.abs()
        phase_l, phase_r = S_l.angle(), S_r.angle()
        cross_spec = S_l * torch.conj(S_r)
        coherence = cross_spec.real / (mag_l * mag_r + 1e-10)
        mask = (coherence < threshold_corr).float()
        freqs = torch.linspace(0, sr/2, n//2+1, device=self.device)
        weight = torch.exp(-freqs / 5000) + 0.5
        mask = mask * weight.unsqueeze(1); mask = torch.clamp(mask, 0, 1)
        if mask.shape[1] > 1:
            k = torch.ones(1, 1, 5, device=self.device)/5
            mask = torch.nn.functional.conv1d(mask.unsqueeze(1), k, padding=2).squeeze(1)
        vec_l = torch.exp(1j * phase_l); vec_r = torch.exp(1j * phase_r)
        corrected_vec_r = (1.0 - mask) * vec_r + mask * vec_l
        corrected_vec_r = corrected_vec_r / (corrected_vec_r.abs() + 1e-10)
        S_r_new = mag_r * corrected_vec_r
        y_l = torch.istft(S_l, n_fft=n, hop_length=h, window=win, length=wav.shape[-1])
        y_r = torch.istft(S_r_new, n_fft=n, hop_length=h, window=win, length=wav.shape[-1])
        return torch.stack([y_l, y_r])

    # ── Stage 4: Stereo Wider (Safe M/S) ──
    def stereo_widen(self, wav, width=0.2):
        if not self.stages.get('stereo_widen') or wav.shape[0] < 2: return wav
        m = (wav[0] + wav[1]) / 2
        s = (wav[0] - wav[1]) / 2
        s = s * (1.0 + width)
        return torch.stack([m + s, m - s])

    # ── Stage 5: Mono-Bass ──
    def mono_bass(self, wav, cutoff=150):
        low = F.lowpass_biquad(F.lowpass_biquad(wav, self.target_sr, cutoff), self.target_sr, cutoff)
        high = wav - low
        if wav.shape[0] >= 2: low = low.mean(dim=0, keepdim=True).expand_as(low)
        return low + high

    # ── Stage 6: Transient Punch ──
    def transient_punch(self, wav, boost_db=4.0):
        if not self.stages.get('transient_punch'): return wav
        sr = self.target_sr
        mono = wav.mean(dim=0) if wav.shape[0]>=2 else wav.squeeze(0)
        fl = int(sr*0.005); hp = fl//2
        if fl < 2: return wav
        p = torch.nn.functional.pad(mono, (fl//2, fl//2))
        e = p.unfold(0, fl, hp).pow(2).mean(dim=-1).sqrt()
        flux = torch.clamp(torch.diff(e, prepend=e[:1]), min=0)
        if flux.max() < 1e-8: return wav
        fn = flux/(flux.max()+1e-8); thr = fn.mean()+1.5*fn.std()
        mask = torch.clamp((fn-thr)/(1.0-thr+1e-8), 0, 1)
        gain = torch.nn.functional.interpolate(mask[None,None,:], size=wav.shape[-1], mode='linear', align_corners=False).squeeze()
        rl = max(int(sr*0.025), 4)
        k = torch.exp(-torch.arange(rl, device=self.device, dtype=torch.float32)/(rl/4))
        k = (k/k.sum())[None,None,:]
        gain = torch.nn.functional.conv1d(gain[None,None,:], k, padding=rl//2).squeeze()[:wav.shape[-1]]
        r = wav*(1.0+gain.unsqueeze(0)*(10**(boost_db/20)-1.0))
        pk = r.abs().max()
        return r*(0.98/pk) if pk > 0.98 else r

    # ── Stage 7: Spectre Restore ──
    def spectre_restore(self, wav):
        if not self.stages.get('spectre_restore'): return wav
        sr = self.target_sr
        stft = torch.stft(wav[0], n_fft=4096, hop_length=1024, window=torch.hann_window(4096).to(self.device), return_complex=True)
        mdb = 20*torch.log10(stft.abs().mean(dim=1)+1e-8)
        freqs = torch.linspace(0, sr/2, mdb.shape[0]).to(self.device)
        v = mdb > (mdb.max()-55)
        co = freqs[v][-1].item() if v.any() else 16000.0
        co = max(12000.0, min(co, 20000.0))
        if co > 19500: return wav
        e1 = F.highpass_biquad(torch.tanh(F.highpass_biquad(wav, sr, co*0.85)*1.8), sr, co*0.9)
        e2 = F.highpass_biquad(torch.tanh(F.highpass_biquad(wav, sr, co)*3.0), sr, co)
        y = wav + e1*0.07 + e2*0.12
        pk = y.abs().max()
        return y*(0.98/pk) if pk > 0.98 else y

    def process_track(self, input_path, output_path, shaper_params=None, phase_control=0.0, stereo_width=0.2):
        if not os.path.exists(input_path): return {'status':'error', 'msg':'File not found', 'stage':'init'}
        try:
            wav, sr = torchaudio.load(input_path)
            wav = self._to_48k(wav.to(self.device), sr)
            if self.stages.get('neural_clean'): wav = self.neural_clean(wav)
            if self.stages.get('spectral_shape'): wav = self.spectral_shape(wav, **(shaper_params or {}))
            if self.stages.get('phase_shape'): wav = self.phase_shape(wav, threshold_corr=phase_control)
            if self.stages.get('stereo_widen'): wav = self.stereo_widen(wav, width=stereo_width)
            if self.stages.get('mono_bass'): wav = self.mono_bass(wav)
            if self.stages.get('transient_punch'): wav = self.transient_punch(wav)
            if self.stages.get('spectre_restore'): wav = self.spectre_restore(wav)
            torchaudio.save(output_path, wav.cpu(), self.target_sr, encoding='PCM_S', bits_per_sample=16)
            return {'status':'ok', 'stage':'Complete'}
        except Exception as e:
            import traceback
            return {'status':'error', 'msg':str(e), 'stage':'processing'}

class MasteringEngine:
    def __init__(self, ref=None):
        self.ref, self.available = ref, False
        if ref and os.path.exists(str(ref)) and HAS_MATCHERING:
            self.available = True; print(f'  🎚️ Mastering ready | Ref: {os.path.basename(ref)}')
        else: print('  ℹ️ Mastering disabled' if not ref else f'  ⚠️ Ref not found: {ref}')
    def master(self, inp, out):
        if not self.available: shutil.copy2(inp,out) if inp!=out else None; return False
        try: mg.process(target=inp, reference=self.ref, results=[mg.pcm16(out)]); return True
        except Exception as e: print(f'  ⚠️ {e}'); shutil.copy2(inp,out) if inp!=out else None; return False
"""

# 3. Update Exec Params
new_exec_start = r"""# === CONFIG ===
INPUT_DIR       = '/kaggle/input/datasets/danieldobles/slavic-songs'
REFERENCE_TRACK = '/kaggle/input/datasets/danieldobles/slavic-songs/REF.flac'
OUTPUT_DIR      = '/kaggle/working/master_organized'
TEMP_DIR        = '/kaggle/working/_temp_restore'
BATCH_SIZE, SET_DUR, N_CLUSTERS = 16, 75*60, 3

STAGES = {'neural_clean':True,'spectral_shape':True,'phase_shape':True,'stereo_widen':True,'mono_bass':True,'transient_punch':True,'spectre_restore':True,'matchering':True}
SHAPER = {'amount': 60, 'speed': 90, 'sensitivity': 30}
PHASE_CONTROL, STEREO_WIDTH = 0.0, 0.2"""

# Apply changes
for cell in nb['cells']:
    if cell.get('id') == 'header':
        cell['source'] = new_header
    elif cell.get('id') == 'restoration':
        lines = new_restoration_code.split('\n')
        cell['source'] = [l + '\n' for l in lines]
    elif cell.get('id') == 'exec':
        old_source = cell['source']
        idx = 0
        for i, line in enumerate(old_source):
            if "print('\\n🔧 Warming Up Pipeline...')" in line:
                idx = i
                break
        
        new_lines = [l + '\n' for l in new_exec_start.split('\n')]
        cell['source'] = new_lines + ["\n"] + old_source[idx:]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)
