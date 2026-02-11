import os
import argparse
import glob
import numpy as np
from joblib import Parallel, delayed
from src.audio_analysis import extract_features
from src.clustering import cluster_audio
from src.file_manager import organize_files
import time

def main():
    parser = argparse.ArgumentParser(description="Spectral Affinity: Group audio files by timbre and rhythm.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing audio files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save organized files.")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters to Create.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of CPU cores to use (-1 for all).")
    parser.add_argument("--clean_names", action="store_true", help="Remove prefixes and IDs from filenames.")
    
    args = parser.parse_args()
    
    # 1. Find Audio Files
    print(f"Scanning {args.input_dir}...")
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.ogg', '*.m4a']
    file_paths = []
    for ext in audio_extensions:
        file_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
        # Case insensitive search would be better but requires more complex globbing or walking
        # For windows, glob is usually case insensitive? Let's assume standard glob.
    
    if not file_paths:
        print("No audio files found! Check the path and extensions.")
        return

    print(f"Found {len(file_paths)} files.")

    # 2. Extract Features (Parallel)
    print(f"Extracting features using {args.n_jobs} jobs...")
    start_time = time.time()
    
    # Parallel execution
    features_list = Parallel(n_jobs=args.n_jobs)(
        delayed(extract_features)(path) for path in file_paths
    )
    
    # Filter out None results (failures)
    valid_features = []
    valid_paths = []
    for f, p in zip(features_list, file_paths):
        if f is not None:
            valid_features.append(f)
            valid_paths.append(p)
            
    if not valid_features:
        print("No features extracted. Exiting.")
        return

    features_matrix = np.array(valid_features)
    print(f"Feature extraction complete in {time.time() - start_time:.2f}s.")

    # 3. Cluster Audio
    print(f"Clustering into {args.n_clusters} groups...")
    labels = cluster_audio(features_matrix, n_clusters=args.n_clusters)
    
    # 4. Organize Files
    print(f"Organizing files into {args.output_dir}...")
    organize_files(valid_paths, labels, args.output_dir, mode='copy', rename=args.clean_names)
    
    print("Done!")

if __name__ == "__main__":
    main()
