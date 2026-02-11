import librosa
import numpy as np
import os

def extract_features(file_path, duration=30):
    """
    Extracts audio features from a file.
    Optimized to load only a specific duration from the center of the track.
    
    Args:
        file_path (str): Path to the audio file.
        duration (int): Duration in seconds to analyze (default: 30).
        
    Returns:
        np.array: Extracted features (MFCCs, Spectral Centroid, Tempo).
    """
    try:
        # Get total duration to find the center
        total_duration = librosa.get_duration(path=file_path)
        
        # Calculate start time to take the segment from the center
        start_time = max(0, (total_duration - duration) // 2)
        
        # Load audio segment
        y, sr = librosa.load(file_path, sr=22050, duration=duration, offset=start_time)
        
        # If the file is shorter than the requested duration, we still process what we have
        if len(y) == 0:
            print(f"Warning: {file_path} is empty or unreadable.")
            return None

        # Extract features
        # 1. MFCCs (Timbre)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)
        
        # 2. Spectral Centroid (Brightness)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_cent_mean = np.mean(spec_cent)
        spec_cent_var = np.var(spec_cent)
        
        # 3. Tempo (Rhythm)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Combine all features into a 1D array
        # 13 MFCC means + 13 MFCC vars + 1 SpecCent Mean + 1 SpecCent Var + 1 Tempo = 29 Features
        features = np.hstack([mfcc_mean, mfcc_var, spec_cent_mean, spec_cent_var, tempo])
        
        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
