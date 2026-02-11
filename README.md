# Spectral Affinity 🎵

Automated audio file grouping based on mathematical features (timbre, tempo, and brightness).

## Features
- **Fast:** Only processes 30 seconds from the center of each track.
- **Parallel:** Uses multi-core processing for feature extraction.
- **Safe:** Copies files to clusters by default (never overwrites originals).
- **Intelligent:** Uses Machine Learning (K-Means) to find patterns.

## Installation
Dependencies should be installed via:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script providing the input directory and the target directory for organization.

```bash
python main.py --input_dir "path/to/your/music" --output_dir "path/to/organized_output" --n_clusters 5
```

### Arguments
- `--input_dir`: Directory containing your MP3, WAV, FLAC, etc.
- `--output_dir`: Where the organized folders (`Cluster_0`, `Cluster_1`, etc.) will be created.
- `--n_clusters`: Into how many groups should the music be divided (default: 5).
- `--n_jobs`: Number of parallel jobs (default: -1, all CPUs).

## How it works
The script creates a "digital fingerprint" for each song using:
- **MFCCs:** Timbre/texture.
- **Spectral Centroid:** Brightness/energy.
- **Tempo:** Rhythm/BPM.

Then, it applies a clustering algorithm to group songs with similar mathematical profiles.
