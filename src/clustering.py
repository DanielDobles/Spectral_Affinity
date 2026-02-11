from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

def cluster_audio(features, n_clusters=5):
    """
    Groups audio files based on extracted features.
    
    Args:
        features (np.array): Matrix of features (N_samples x M_features).
        n_clusters (int): Number of groups to form (default: 5).
        
    Returns:
        np.array: Array of integer labels for each sample.
    """
    # 1. Scale Features
    # Features have different scales (Tempo is around 120, MFCCs are small).
    # StandardScaler ensures each feature contributes equally.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # 2. Cluster using KMeans
    # KMeans is simple and effective for this task.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    
    return labels
