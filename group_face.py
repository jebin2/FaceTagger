import numpy as np
import dlib
import hdbscan
import face_recognition
from pathlib import Path
import os
from tqdm import tqdm
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
    if len(encodings_list) == 0:
        return np.array([])
    
    encodings = np.array(encodings_list)
    
    scaler = StandardScaler()
    encodings_scaled = scaler.fit_transform(encodings)
    
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
    
    processed_encodings = preprocess_encodings(encodings)
    if processed_encodings.size == 0:
        return np.array([], dtype=int)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    cluster_labels = clusterer.fit_predict(processed_encodings)

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
    
    if n_clusters is None:
        n_clusters = max(2, min(20, len(encodings) // 8))
        if n_clusters >= len(encodings):
            n_clusters = len(encodings) - 1
        print(f"Estimating number of clusters: {n_clusters}")

    if n_clusters <= 1:
        return np.array([0] * len(encodings), dtype=int)
    
    processed_encodings = preprocess_encodings(encodings)
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
    
    quality_scores = []
    centroid = np.mean(encodings_np, axis=0)
    for encoding in tqdm(encodings_np, desc="Calculating quality scores"):
        quality = 1.0 / (1.0 + np.linalg.norm(encoding - centroid))
        quality_scores.append(quality)
    
    quality_scores = np.array(quality_scores)
    quality_order = np.argsort(quality_scores)[::-1]
    
    for idx in tqdm(quality_order, desc="Adaptive clustering"):
        if labels[idx] != -1:
            continue
        face_quality = quality_scores[idx]
        adaptive_tolerance = base_tolerance * (1.0 + (1.0 - face_quality) * 0.3)
        
        labels[idx] = current_label
        distances = face_recognition.face_distance(encodings_np, encodings_np[idx])
        
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
        
    if n_clusters is None:
        n_clusters = max(2, min(15, len(encodings) // 5))
        if n_clusters >= len(encodings):
            n_clusters = len(encodings) - 1
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
    clustering = AffinityPropagation(damping=0.75, random_state=42)
    labels = clustering.fit_predict(processed_encodings)
    return labels

def cluster_with_chinese_whispers(encodings, tolerance=0.5):
    """Groups faces using the Chinese Whispers algorithm from dlib."""
    print("Clustering with Chinese Whispers...")
    if len(encodings) < 2:
        return [0] * len(encodings)
        
    dlib_encodings = [dlib.vector(enc) for enc in tqdm(encodings, desc="Converting to dlib vectors")]
    labels = dlib.chinese_whispers_clustering(dlib_encodings, tolerance)
    return labels

def compare_and_group_faces_with_fr(input_folder, tolerance=0.9):
    """
    Compare all faces in a folder using face_recognition + cosine similarity.
    Returns a list of labels (person IDs) in the same order as the images in the folder.
    """
    input_folder = Path(input_folder)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    encodings = []
    for img_file in tqdm(image_files, desc="Loading images & extracting encodings"):
        img_path = input_folder / img_file
        img = face_recognition.load_image_file(img_path)
        faces = face_recognition.face_encodings(img)
        if faces:
            encodings.append(faces[0])
        else:
            encodings.append(None)

    labels = [-1] * len(image_files)
    current_label = 0

    for i in tqdm(range(len(image_files)), desc="Comparing faces (outer loop)"):
        if encodings[i] is None or labels[i] != -1:
            continue

        labels[i] = current_label
        for j in tqdm(range(i + 1, len(image_files)), desc=f"Comparing with person_{current_label:04d}", leave=False):
            if encodings[j] is None or labels[j] != -1:
                continue

            match = face_recognition.compare_faces([encodings[i]], encodings[j], tolerance=tolerance)[0]

            if match:
                labels[j] = current_label

        current_label += 1

    return labels
