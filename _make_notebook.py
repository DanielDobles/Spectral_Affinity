"""Generate Spectral_Affinity_Master.ipynb with valid JSON."""
import json, os

cells = []

# ── Markdown Header ──
cells.append({
    "cell_type": "markdown", "id": "header", "metadata": {},
    "source": [
        "# Spectral Affinity Master v6.3\n",
        "### 9-Stage Ultimate Suno Master Pipeline\n",
        "\n",
        "| Stage | Module | Effect |\n",
        "|-------|--------|--------|\n",
        "| 1 | Neural Clean | DeepFilterNet 3 |\n",
        "| 2 | Spectral Shaper | STFT Stabilizer |\n",
        "| 3 | Phase Shaper | Phase coherence |\n",
        "| 4 | Stereo Wider | M/S + Haas |\n",
        "| 5 | Mono-Bass | Sub mono |\n",
        "| 6 | Transient Punch | Envelope Mask |\n",
        "| 7 | Spectre Restore | HF Exciter |\n",
        "| 8 | Affinity Grouping | MERT clustering |\n",
        "| 9 | Mastering Match | Matchering |"
    ]
})

# ── Setup Cell ──
cells.append({
    "cell_type": "code", "id": "setup", "metadata": {},
    "execution_count": None, "outputs": [],
    "source": [
        "import os, warnings, torch, torchaudio, glob, shutil, gc\n",
        "from tqdm.auto import tqdm\n",
        "from IPython.display import HTML, display\n",
        "from ultimate_pipeline import UltimateSunoMaster, MasteringEngine, build_track_name\n",
        "\n",
        "warnings.filterwarnings('ignore')\n",
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
        "\n",
        "try:\n",
        "    import matchering\n",
        "except Exception:\n",
        "    !pip install -q deepfilternet matchering\n",
        "\n",
        "print('GPU:', torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU')\n",
    ]
})

# ── Config Cell ──
cells.append({
    "cell_type": "code", "id": "config", "metadata": {},
    "execution_count": None, "outputs": [],
    "source": [
        "# === CONFIG v6.3 ===\n",
        "INPUT_DIR       = '/kaggle/input/datasets/danieldobles/slavic-songs'\n",
        "REFERENCE_TRACK = '/kaggle/input/datasets/danieldobles/slavic-songs/REF.flac'\n",
        "OUTPUT_DIR      = '/kaggle/working/master_organized'\n",
        "TEMP_DIR        = '/kaggle/working/_temp_restore'\n",
        "\n",
        "# Stabilizer Panel\n",
        "SHAPER        = {'amount': 60, 'speed': 90, 'sensitivity': 30}\n",
        "PHASE_CONTROL = -0.3    # -1 tight (linear) to +1 warm (analog)\n",
        "STEREO_WIDTH  = 0.3     # subtle but powerful\n",
        "\n",
        "STAGES = {\n",
        "    'neural_clean':    True,\n",
        "    'spectral_shape':  True,\n",
        "    'phase_shape':     True,\n",
        "    'stereo_widen':    True,\n",
        "    'mono_bass':       True,\n",
        "    'transient_punch': True,\n",
        "    'spectre_restore': True,\n",
        "}\n",
        "\n",
        "print('Init engines...')\n",
        "restorer  = UltimateSunoMaster(device=device, stages=STAGES)\n",
        "mastering = MasteringEngine(reference_path=REFERENCE_TRACK)\n",
    ]
})

# ── Execution Cell ──
cells.append({
    "cell_type": "code", "id": "run", "metadata": {},
    "execution_count": None, "outputs": [],
    "source": [
        "exts = ['**/*.flac', '**/*.mp3', '**/*.wav', '**/*.m4a']\n",
        "paths = []\n",
        "for e in exts:\n",
        "    paths.extend(glob.glob(os.path.join(INPUT_DIR, e), recursive=True))\n",
        "paths = sorted(set(paths))\n",
        "\n",
        "if not paths:\n",
        "    print('No tracks found.')\n",
        "else:\n",
        "    os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
        "    os.makedirs(TEMP_DIR, exist_ok=True)\n",
        "    print('Processing', len(paths), 'tracks...')\n",
        "\n",
        "    for i, p in enumerate(tqdm(paths)):\n",
        "        out_name = build_track_name(i + 1, p)\n",
        "        final_path = os.path.join(OUTPUT_DIR, out_name)\n",
        "        temp_path  = os.path.join(TEMP_DIR, 'tmp_' + str(i) + '.wav')\n",
        "\n",
        "        ok = restorer.process_track(\n",
        "            p, temp_path,\n",
        "            shaper_params=SHAPER,\n",
        "            phase_control=PHASE_CONTROL,\n",
        "            stereo_width=STEREO_WIDTH\n",
        "        )\n",
        "        if ok:\n",
        "            mastering.master(temp_path, final_path)\n",
        "            if os.path.exists(temp_path):\n",
        "                os.remove(temp_path)\n",
        "        else:\n",
        "            shutil.copy2(p, final_path)\n",
        "\n",
        "        if i % 10 == 0:\n",
        "            gc.collect()\n",
        "            torch.cuda.empty_cache()\n",
        "\n",
        "    shutil.rmtree(TEMP_DIR, ignore_errors=True)\n",
        "    !zip -0 -rq Master_v6_3.zip master_organized\n",
        "    print('Done!')\n",
        "    display(HTML('<h3><a href=\"Master_v6_3.zip\">Download Master v6.3</a></h3>'))\n",
    ]
})

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"}
    },
    "cells": cells
}

out_path = os.path.join(os.path.dirname(__file__), "Spectral_Affinity_Master.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"OK: {out_path}")
