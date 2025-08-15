import numpy as np
import dlib
import hdbscan
import face_recognition

# --- All imports moved to the top of the file to prevent circular import errors ---
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (
    SpectralClustering, 
    AgglomerativeClustering, 
    AffinityPropagation
)
from sklearn.neighbors import kneighbors_graph


def preprocess_encodings(encodings_list):
    """Scales and applies PCA to encodings for better clustering performance."""
    # --- CORRECTED CHECK: Only check the length. ---
    if len(encodings_list) == 0:
        return np.array([])
    
    encodings = np.array(encodings_list)
    
    # Scale data to have zero mean and unit variance
    scaler = StandardScaler()
    encodings_scaled = scaler.fit_transform(encodings)
    
    # Reduce dimensions to handle noise (optional but recommended)
    # n_components should be less than the number of samples and features
    num_components = min(128, len(encodings) - 1) if len(encodings) > 1 else 1
    if num_components > 0:
        pca = PCA(n_components=num_components)
        encodings_pca = pca.fit_transform(encodings_scaled)
        return encodings_pca
    return encodings_scaled

def cluster_with_hdbscan(encodings, min_cluster_size=5, min_samples=3):
    """
    Advanced hierarchical density-based clustering - often superior to DBSCAN for faces.
    """
    print("Clustering with HDBSCAN...")
    if len(encodings) < 2:
        return np.array([0] * len(encodings), dtype=int)
    
    # Preprocess encodings
    processed_encodings = preprocess_encodings(encodings)
    if processed_encodings.size == 0:
        return np.array([], dtype=int)
    
    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'  # Excess of Mass method
    )
    
    cluster_labels = clusterer.fit_predict(processed_encodings)
    
    # Get cluster probabilities for quality assessment
    probabilities = clusterer.probabilities_
    print(f"Found {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters")
    print(f"Average cluster probability: {probabilities.mean():.3f}")
    
    return cluster_labels

def cluster_with_spectral(encodings, n_clusters=None, n_neighbors=10):
    """
    Spectral clustering using similarity graph approach.
    """
    print("Clustering with Spectral Clustering...")
    if len(encodings) < 2:
        return np.array([0] * len(encodings), dtype=int)
    
    # Estimate number of clusters if not provided
    if n_clusters is None:
        # Ensure we don't request more clusters than samples
        n_clusters = max(2, min(20, len(encodings) // 8))
        if n_clusters >= len(encodings):
             n_clusters = len(encodings) -1
        print(f"Estimating number of clusters: {n_clusters}")

    if n_clusters <= 1:
        return np.array([0] * len(encodings), dtype=int)
    
    processed_encodings = preprocess_encodings(encodings)
    
    # Create similarity graph
    connectivity = kneighbors_graph(
        processed_encodings, 
        n_neighbors=n_neighbors, 
        mode='connectivity',
        include_self=True
    )
    
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        random_state=42,
        n_jobs=-1
    )
    
    cluster_labels = spectral.fit_predict(processed_encodings)
    return cluster_labels

def cluster_with_adaptive_similarity(encodings, base_tolerance=0.55, quality_threshold=0.8):
    """
    Enhanced similarity clustering with adaptive thresholds.
    """
    print("Clustering with Adaptive Similarity...")
    if len(encodings) < 2:
        return [0] * len(encodings)
    
    encodings_np = np.array(encodings)
    n_faces = len(encodings_np)
    labels = [-1] * n_faces
    current_label = 0
    
    # Calculate face quality scores (based on embedding confidence)
    quality_scores = []
    centroid = np.mean(encodings_np, axis=0)
    for encoding in encodings_np:
        # Simple quality metric: inverse of distance to centroid
        quality = 1.0 / (1.0 + np.linalg.norm(encoding - centroid))
        quality_scores.append(quality)
    
    quality_scores = np.array(quality_scores)
    
    # Sort faces by quality (high-quality faces first)
    quality_order = np.argsort(quality_scores)[::-1]
    
    for idx in quality_order:
        if labels[idx] != -1:
            continue
        
        # Adaptive threshold based on face quality
        face_quality = quality_scores[idx]
        adaptive_tolerance = base_tolerance * (1.0 + (1.0 - face_quality) * 0.3)
        
        labels[idx] = current_label
        distances = face_recognition.face_distance(encodings_np, encodings_np[idx])
        
        # Find all similar faces with adaptive threshold
        similar_indices = np.where(distances <= adaptive_tolerance)[0]
        
        for similar_idx in similar_indices:
            if labels[similar_idx] == -1:
                labels[similar_idx] = current_label
        
        current_label += 1
    
    return labels

def cluster_with_agglomerative(encodings, n_clusters=None):
    """Groups faces using Agglomerative Hierarchical Clustering."""
    print("Clustering with Agglomerative Clustering...")
    if len(encodings) < 2:
        return np.array([0] * len(encodings), dtype=int)
        
    # If n_clusters is not specified, we can try to estimate it
    if n_clusters is None:
        n_clusters = max(2, min(15, len(encodings) // 5))
        if n_clusters >= len(encodings):
             n_clusters = len(encodings) -1
        print(f"Estimating number of clusters: {n_clusters}")

    if n_clusters <= 1:
        return np.array([0] * len(encodings), dtype=int)

    processed_encodings = preprocess_encodings(encodings)
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(processed_encodings)
    return labels

def cluster_with_affinity_propagation(encodings):
    """Groups faces using the Affinity Propagation algorithm."""
    print("Clustering with Affinity Propagation...")
    if len(encodings) < 2:
        return np.array([0] * len(encodings), dtype=int)

    processed_encodings = preprocess_encodings(encodings)
    
    # The 'damping' parameter can help with convergence.
    clustering = AffinityPropagation(damping=0.75, random_state=42)
    labels = clustering.fit_predict(processed_encodings)
    return labels

def cluster_with_chinese_whispers(encodings, tolerance=0.5):
    """Groups faces using the Chinese Whispers algorithm from dlib."""
    print("Clustering with Chinese Whispers...")
    if len(encodings) < 2:
        return [0] * len(encodings)
        
    # dlib expects a list of face descriptor objects
    dlib_encodings = [dlib.vector(enc) for enc in encodings]
    
    # The second parameter is the tolerance.
    labels = dlib.chinese_whispers_clustering(dlib_encodings, tolerance)
    
    return labels