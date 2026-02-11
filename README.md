# Spectral_Affinity
Computational Acoustic Taxonomy &amp; Unsupervised Audio Clustering

Abstract: Spectral Affinity is a Python-based analytical framework designed to deconstruct and categorize audio datasets based on their latent mathematical properties. Rather than relying on subjective metadata (genre tags or artist names), this system performs a morphological analysis of the raw audio signal.

By leveraging Librosa for signal processing and Scikit-learn for machine learning, the algorithm extracts high-dimensional acoustic features—specifically Mel-Frequency Cepstral Coefficients (MFCCs) for timbre, Spectral Centroids for brightness, and Tempo (BPM) for rhythmic structure.

These features are projected into a normalized vector space where an unsupervised K-Means Clustering algorithm identifies inherent "affinities" or structural similarities between tracks. The result is a mathematically coherent organization of a music library, grouping compositions by their sonic texture and dynamic signature rather than arbitrary labels.

Key Features:

Signal Decomposition: Utilizes Fast Fourier Transform (FFT) derivatives to isolate specific audio characteristics.

Vector Space Modeling: Converts audio waves into numerical vectors for precise mathematical comparison.

Local Compute Optimization: Engineered to run efficiently on standard CPU architectures using parallel processing (joblib), eliminating the need for GPU acceleration or cloud dependencies.

Automated Taxonomy: Sorts hundreds of tracks into distinct, cohesive sonic clusters without human intervention.

Technical Stack:

Language: Python 3.x

Audio Analysis: Librosa

Machine Learning: Scikit-learn (K-Means / Nearest Neighbors)

Data Handling: NumPy, Pandas
