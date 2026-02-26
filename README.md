# 🌌 Spectral Affinity v3.0 — Music-CLAP Edition

### *The Ultimate AI Librarian for Semantic Clustering & Camelot-based Harmonic Flow*

---

**Spectral Affinity** is a read-only AI Librarian designed to analyze, classify, and organize your music collection without ever altering the original audio samples. It leverages state-of-the-art neural networks to understand the "vibe" and harmonic structure of your tracks, enabling seamless DJ-grade playlist generation and semantic organization.

## 🚀 Key Features

- 🧠 **Neural Key Detection:** Uses `nnAudio CQT1992v2` to convert Chromagrams into precise Camelot notation.
- 🎵 **Semantic Embeddings:** Powered by `Microsoft Music-CLAP` for deep understanding of vibe, genre, and mood.
- 🥁 **Rhythm Analysis:** Employs `librosa` for accurate BPM tracking across your library.
- 🔀 **Semantic Clustering:** Groups tracks by acoustic vibe using `KMeans` in the Music-CLAP embedding space.
- 🎛️ **Harmonic Sequencing:** Implements a Camelot Wheel Engine for DJ-grade, BPM-ascending playlist flow.
- 📦 **Smart Export:** Automatically renames and organizes files into ZIP archives for easy download and use.

## 🛠️ Technology Stack

| Module | Technology | Role |
|--------|-----------|------|
| **Embedding Engine** | Microsoft Music-CLAP | Vibe / genre / mood vectors |
| **Spectral Analysis** | nnAudio | Neural Key & Pitch detection |
| **Rhythm Analysis** | librosa | Beat tracking and BPM calculation |
| **Clustering** | scikit-learn (KMeans) | High-dimensional semantic grouping |
| **File Management** | shutil + zipfile | Non-destructive renaming & packaging |

## 📦 Installation & Setup

This project is optimized for environments with GPU support (like Kaggle or local NVIDIA setups).

### Dependencies

To run the notebook locally, ensure you have Python 3.10+ and install the following:

```bash
pip install numpy==1.26.4
pip install nnAudio>=0.3.3 msclap>=1.3.3 librosa>=0.10.0 scikit-learn>=1.3.0 tqdm soundfile
```

*Note: `numpy==1.26.4` is strictly required for compatibility with certain spectral analysis modules.*

## 📖 How it Works

1. **Ingestion:** Load your audio files into the environment.
2. **Analysis:** The system extracts BPM, Key (Camelot), and Semantic Embeddings (CLAP).
3. **Clustering:** KMeans identifies groups of tracks that "feel" similar, regardless of metadata.
4. **Sequencing:** The engine sorts tracks within clusters to ensure harmonic transitions (following the Camelot Wheel).
5. **Organization:** Files are renamed with their new metadata (e.g., `12A - 124BPM - TrackName.mp3`) and bundled for export.

---

> **⚠️ Disclaimer:** This tool is strictly **read-only**. It analyzes, classifies, renames, and copies your files — it never modifies the internal audio data of your original samples.
