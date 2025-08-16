import numpy as np
from pathlib import Path
import logging
import logging.handlers
from collections import defaultdict
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import shutil
from typing import List, Dict, Tuple, Optional
import sys
import time
from datetime import datetime


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional log file path. If None, logs to console only.
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB max, 5 backups
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logging.info(f"Logging to file: {log_file}")
    
    return logger


class ImprovedRegrouper:
    """
    Enhanced face regrouping system with multiple strategies and comprehensive logging.
    """
    
    def __init__(self, detect_manager, similarity_threshold=0.75, log_file=None):
        """
        Initialize the regrouper with comprehensive logging.
        
        Args:
            detect_manager: Face detection/embedding manager
            similarity_threshold: Base similarity threshold for face matching
            log_file: Optional log file path
        """
        self.detect_manager = detect_manager
        self.similarity_threshold = similarity_threshold
        
        # Setup logging
        self.logger = setup_logging(logging.INFO, log_file)
        self.logger.info("ImprovedRegrouper initialized")
        self.logger.info(f"Similarity threshold: {similarity_threshold}")
        
        # Statistics tracking
        self.stats = {
            'total_comparisons': 0,
            'successful_merges': 0,
            'failed_comparisons': 0,
            'processing_time': 0
        }
    
    def regroup_with_multiple_representatives(self, output_dir: Path, 
                                           num_representatives=5,
                                           consensus_threshold=0.6):
        """
        Use multiple representative faces per folder for more robust comparison.
        
        Args:
            output_dir: Directory containing person_XXX folders
            num_representatives: Number of faces to use as representatives
            consensus_threshold: Fraction of comparisons that must agree to merge
        """
        start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("STARTING REGROUPING WITH MULTIPLE REPRESENTATIVES")
        self.logger.info(f"Directory: {output_dir}")
        self.logger.info(f"Representatives per folder: {num_representatives}")
        self.logger.info(f"Consensus threshold: {consensus_threshold}")
        self.logger.info("=" * 60)
        
        person_folders = sorted([f for f in output_dir.iterdir() 
                               if f.is_dir() and f.name.startswith("person_")])
        
        if len(person_folders) < 2:
            self.logger.warning(f"Only {len(person_folders)} folders found. Regrouping requires at least 2 folders.")
            return
        
        self.logger.info(f"Found {len(person_folders)} person folders to process")
        
        # Get multiple representatives per folder
        folder_representatives = {}
        for i, folder in enumerate(person_folders):
            self.logger.info(f"Processing folder {i+1}/{len(person_folders)}: {folder.name}")
            representatives = self._get_multiple_representatives(folder, num_representatives)
            if representatives:
                folder_representatives[folder.name] = representatives
                self.logger.info(f"Found {len(representatives)} representatives in {folder.name}")
            else:
                self.logger.warning(f"No valid representatives found in {folder.name}")
        
        if len(folder_representatives) < 2:
            self.logger.warning("Not enough folders with valid representatives for regrouping")
            return
        
        # Compare folders using consensus voting
        self.logger.info(f"Starting consensus-based comparison of {len(folder_representatives)} folders")
        merge_mapping = self._find_merges_with_consensus(
            folder_representatives, consensus_threshold
        )
        
        # Execute merges
        if merge_mapping:
            self.logger.info(f"Executing {len(merge_mapping)} merges")
            self._execute_merges(output_dir, merge_mapping)
        else:
            self.logger.info("No merges identified")
        
        processing_time = time.time() - start_time
        self.stats['processing_time'] = processing_time
        self.logger.info("=" * 60)
        self.logger.info("REGROUPING COMPLETED")
        self.logger.info(f"Processing time: {processing_time:.2f} seconds")
        self.logger.info(f"Successful merges: {self.stats['successful_merges']}")
        self.logger.info(f"Total comparisons: {self.stats['total_comparisons']}")
        self.logger.info(f"Failed comparisons: {self.stats['failed_comparisons']}")
        self.logger.info("=" * 60)
        
    def regroup_with_quality_scoring(self, output_dir: Path):
        """
        Use quality-based selection of representative faces.
        Consider face size, sharpness, frontality, and lighting.
        """
        start_time = time.time()
        self.logger.info("STARTING QUALITY-BASED REGROUPING")
        
        person_folders = sorted([f for f in output_dir.iterdir() 
                               if f.is_dir() and f.name.startswith("person_")])
        
        self.logger.info(f"Processing {len(person_folders)} folders with quality scoring")
        
        folder_representatives = {}
        for folder in person_folders:
            self.logger.debug(f"Analyzing quality in {folder.name}")
            best_face = self._get_highest_quality_face(folder)
            if best_face:
                folder_representatives[folder.name] = [best_face]
                self.logger.debug(f"Best quality face in {folder.name}: {best_face.name}")
        
        merge_mapping = self._find_merges_simple(folder_representatives)
        self._execute_merges(output_dir, merge_mapping)
        
        self.logger.info(f"Quality-based regrouping completed in {time.time() - start_time:.2f} seconds")
    
    def regroup_with_adaptive_threshold(self, output_dir: Path):
        """
        Use adaptive thresholding based on cluster sizes and quality.
        """
        start_time = time.time()
        self.logger.info("STARTING ADAPTIVE THRESHOLD REGROUPING")
        
        person_folders = sorted([f for f in output_dir.iterdir() 
                               if f.is_dir() and f.name.startswith("person_")])
        
        # Calculate adaptive thresholds based on folder characteristics
        self.logger.info("Calculating adaptive thresholds for each folder")
        folder_info = {}
        for folder in person_folders:
            face_files = list(folder.glob("*.jpg"))
            avg_quality = self._estimate_folder_quality(folder)
            adaptive_threshold = self._calculate_adaptive_threshold(len(face_files), avg_quality)
            
            folder_info[folder.name] = {
                'size': len(face_files),
                'quality': avg_quality,
                'threshold': adaptive_threshold
            }
            
            self.logger.debug(f"{folder.name}: size={len(face_files)}, "
                            f"quality={avg_quality:.3f}, threshold={adaptive_threshold:.3f}")
        
        # Get representatives
        folder_representatives = {}
        for folder in person_folders:
            representatives = self._get_multiple_representatives(folder, 3)
            if representatives:
                folder_representatives[folder.name] = representatives
        
        # Find merges with adaptive thresholds
        merge_mapping = self._find_merges_adaptive(folder_representatives, folder_info)
        self._execute_merges(output_dir, merge_mapping)
        
        self.logger.info(f"Adaptive threshold regrouping completed in {time.time() - start_time:.2f} seconds")
    
    def regroup_with_embedding_clustering(self, output_dir: Path):
        """
        Re-cluster all faces using their embeddings to find better groupings.
        This essentially re-does clustering but with post-processing insights.
        """
        start_time = time.time()
        self.logger.info("STARTING EMBEDDING-BASED RE-CLUSTERING")
        
        # Collect all face embeddings with their current assignments
        all_embeddings = []
        face_to_folder = {}
        folder_faces = defaultdict(list)
        
        person_folders = sorted([f for f in output_dir.iterdir() 
                               if f.is_dir() and f.name.startswith("person_")])
        
        self.logger.info(f"Collecting embeddings from {len(person_folders)} folders")
        
        total_faces = 0
        for folder in person_folders:
            folder_face_count = 0
            for face_file in folder.glob("*.jpg"):
                # Get embedding for this face
                embedding = self._get_face_embedding(face_file)
                if embedding is not None:
                    all_embeddings.append(embedding)
                    face_to_folder[len(all_embeddings)-1] = folder.name
                    folder_faces[folder.name].append((face_file, len(all_embeddings)-1))
                    folder_face_count += 1
                    total_faces += 1
                else:
                    self.logger.warning(f"Could not get embedding for {face_file}")
            
            self.logger.debug(f"Collected {folder_face_count} embeddings from {folder.name}")
        
        if not all_embeddings:
            self.logger.error("No valid embeddings collected. Cannot proceed with re-clustering.")
            return
        
        self.logger.info(f"Total embeddings collected: {len(all_embeddings)}")
        
        # Re-cluster using DBSCAN with optimized parameters
        embeddings_array = np.array(all_embeddings)
        
        # Use cosine distance for face embeddings
        self.logger.info("Calculating cosine distances between embeddings")
        distances = 1 - cosine_similarity(embeddings_array)
        
        # Adaptive eps based on embedding distribution
        eps = self._calculate_optimal_eps(distances)
        self.logger.info(f"Using DBSCAN with eps={eps:.3f}")
        
        clustering = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
        new_labels = clustering.fit_predict(distances)
        
        unique_labels = set(new_labels)
        n_clusters = len(unique_labels) - (1 if -1 in new_labels else 0)
        n_noise = list(new_labels).count(-1)
        
        self.logger.info(f"DBSCAN results: {n_clusters} clusters, {n_noise} noise points")
        
        # Create new grouping based on clustering results
        self._create_new_groupings(output_dir, face_to_folder, folder_faces, new_labels)
        
        self.logger.info(f"Embedding re-clustering completed in {time.time() - start_time:.2f} seconds")
    
    def regroup_with_temporal_context(self, output_dir: Path):
        """
        Use temporal information (frame numbers/timestamps) to improve regrouping.
        Faces appearing in similar timeframes are more likely to be the same person.
        """
        start_time = time.time()
        self.logger.info("STARTING TEMPORAL CONTEXT REGROUPING")
        
        person_folders = sorted([f for f in output_dir.iterdir() 
                               if f.is_dir() and f.name.startswith("person_")])
        
        # Extract temporal information from filenames
        self.logger.info("Extracting temporal information from filenames")
        folder_temporal_info = {}
        for folder in person_folders:
            temporal_data = self._extract_temporal_info(folder)
            if temporal_data and (temporal_data['timestamps'] or temporal_data['frame_numbers']):
                folder_temporal_info[folder.name] = temporal_data
                self.logger.debug(f"{folder.name}: {len(temporal_data['timestamps'])} timestamps, "
                               f"{len(temporal_data['frame_numbers'])} frame numbers")
            else:
                self.logger.warning(f"No temporal information found in {folder.name}")
        
        if len(folder_temporal_info) < 2:
            self.logger.warning("Not enough folders with temporal information for regrouping")
            return
        
        # Find temporal overlaps and use them to weight similarity scores
        merge_mapping = self._find_merges_with_temporal_weighting(
            folder_temporal_info, output_dir
        )
        
        self._execute_merges(output_dir, merge_mapping)
        
        self.logger.info(f"Temporal context regrouping completed in {time.time() - start_time:.2f} seconds")
    
    def _get_multiple_representatives(self, folder_path: Path, 
                                    num_representatives: int) -> List[Path]:
        """Get multiple representative faces using quality scoring."""
        face_files = list(folder_path.glob("*.jpg"))
        if not face_files:
            self.logger.warning(f"No face files found in {folder_path}")
            return []
        
        self.logger.debug(f"Scoring {len(face_files)} faces in {folder_path.name}")
        
        # Score each face
        face_scores = []
        for face_file in face_files:
            score = self._calculate_face_quality_score(face_file)
            face_scores.append((face_file, score))
        
        # Sort by score and take top representatives
        face_scores.sort(key=lambda x: x[1], reverse=True)
        representatives = [fs for fs in face_scores[:num_representatives]]
        
        self.logger.debug(f"Selected {len(representatives)} representatives from {folder_path.name}")
        if self.logger.isEnabledFor(logging.DEBUG):
            for i, (face_file, score) in enumerate(face_scores[:num_representatives]):
                self.logger.debug(f"  Rep {i+1}: {face_file.name} (score: {score:.3f})")
        
        return representatives
    
    def _calculate_face_quality_score(self, face_file: Path) -> float:
        """
        Calculate comprehensive quality score for a face.
        Considers: size, sharpness, contrast, face detection confidence.
        """
        try:
            img = cv2.imread(str(face_file))
            if img is None:
                self.logger.warning(f"Could not load image: {face_file}")
                return 0.0
            
            # Size score (larger faces generally better)
            height, width = img.shape[:2]
            size_score = min(1.0, (width * height) / (100 * 100))  # Normalize to 100x100
            
            # Sharpness score using Laplacian variance
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 1000.0)  # Normalize
            
            # Contrast score
            contrast_score = gray.std() / 127.5  # Normalize to 0-1
            
            # Brightness consistency (avoid over/under exposed)
            mean_brightness = gray.mean()
            brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5
            
            # File size as proxy for quality (JPEG compression artifacts)
            file_size = face_file.stat().st_size
            size_score_file = min(1.0, file_size / 50000)  # Normalize to ~50KB
            
            # Weighted combination
            total_score = (size_score * 0.3 + 
                          sharpness_score * 0.3 + 
                          contrast_score * 0.2 + 
                          brightness_score * 0.1 + 
                          size_score_file * 0.1)
            
            return total_score
            
        except Exception as e:
            self.logger.error(f"Error calculating quality for {face_file}: {e}")
            return 0.0
    
    def _get_highest_quality_face(self, folder_path: Path) -> Optional[Path]:
        """Get single highest quality face from folder."""
        representatives = self._get_multiple_representatives(folder_path, 1)
        return representatives[0] if representatives else None
    
    def _find_merges_with_consensus(self, folder_representatives: Dict, 
                                  consensus_threshold: float) -> Dict[str, str]:
        """Find merges using consensus voting from multiple representatives."""
        self.logger.info(f"Finding merges with consensus threshold: {consensus_threshold}")
        
        merge_mapping = {}
        folder_names = sorted(folder_representatives.keys())
        merged_folders = set()
        
        total_pairs = len(folder_names) * (len(folder_names) - 1) // 2
        current_pair = 0
        
        for i, current_folder in enumerate(folder_names):
            if current_folder in merged_folders:
                continue
            
            current_faces = folder_representatives[current_folder]
            
            for j in range(i + 1, len(folder_names)):
                current_pair += 1
                compare_folder = folder_names[j]
                
                if current_pair % 10 == 0:
                    self.logger.info(f"Processing pair {current_pair}/{total_pairs}: "
                                   f"{current_folder} vs {compare_folder}")
                
                if compare_folder in merged_folders:
                    continue
                
                compare_faces = folder_representatives[compare_folder]
                
                # Calculate similarity between all pairs
                similarities = []
                for curr_face in current_faces:
                    for comp_face in compare_faces:
                        sim = self.detect_manager.compute_similarity(str(curr_face[0]), str(comp_face[0]))
                        self.stats['total_comparisons'] += 1
                        if sim is not None:
                            similarities.append(sim)
                        else:
                            self.stats['failed_comparisons'] += 1
                
                if similarities:
                    # Check consensus: what fraction of comparisons exceed threshold
                    above_threshold = sum(1 for s in similarities if s >= self.similarity_threshold)
                    consensus = above_threshold / len(similarities)
                    avg_similarity = np.mean(similarities)
                    max_similarity = np.max(similarities)
                    
                    self.logger.debug(f"{current_folder} vs {compare_folder}: "
                                    f"consensus={consensus:.3f}, avg_sim={avg_similarity:.3f}, "
                                    f"max_sim={max_similarity:.3f}")
                    
                    if consensus >= consensus_threshold:
                        self.logger.info(f"MERGE IDENTIFIED: {compare_folder} -> {current_folder} "
                                       f"(consensus: {consensus:.3f}, avg_sim: {avg_similarity:.3f}, "
                                       f"max_sim: {max_similarity:.3f})")
                        merge_mapping[compare_folder] = current_folder
                        merged_folders.add(compare_folder)
                        self.stats['successful_merges'] += 1
        
        self.logger.info(f"Consensus analysis complete. Found {len(merge_mapping)} merges.")
        return merge_mapping
    
    def _find_merges_simple(self, folder_representatives: Dict) -> Dict[str, str]:
        """Simple pairwise comparison for single representatives."""
        self.logger.info("Finding merges with simple pairwise comparison")
        
        merge_mapping = {}
        folder_names = sorted(folder_representatives.keys())
        merged_folders = set()
        
        for i, current_folder in enumerate(folder_names):
            if current_folder in merged_folders:
                continue
            
            current_face = folder_representatives[current_folder][0]
            
            for j in range(i + 1, len(folder_names)):
                compare_folder = folder_names[j]
                if compare_folder in merged_folders:
                    continue
                
                compare_face = folder_representatives[compare_folder][0]
                similarity = self.detect_manager.compute_similarity(str(current_face[0]), str(compare_face[0]))
                self.stats['total_comparisons'] += 1
                
                if similarity is not None and similarity >= self.similarity_threshold:
                    self.logger.info(f"MERGE: {compare_folder} -> {current_folder} "
                                   f"(similarity: {similarity:.3f})")
                    merge_mapping[compare_folder] = current_folder
                    merged_folders.add(compare_folder)
                    self.stats['successful_merges'] += 1
                elif similarity is None:
                    self.stats['failed_comparisons'] += 1
        
        return merge_mapping
    
    def _calculate_face_similarity(self, face1_path: Path, face2_path: Path) -> Optional[float]:
        """Calculate similarity between two face images."""
        try:
            # Load images
            img1 = cv2.imread(str(face1_path))
            img2 = cv2.imread(str(face2_path))
            
            if img1 is None or img2 is None:
                self.logger.warning(f"Could not load images: {face1_path} or {face2_path}")
                return None
            
            # Get embeddings using the detect manager
            embedding1 = self.detect_manager.get_embedding(img1)
            embedding2 = self.detect_manager.get_embedding(img2)
            
            if embedding1 is None or embedding2 is None:
                self.logger.warning(f"Could not get embeddings for {face1_path} or {face2_path}")
                return None
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity between {face1_path} and {face2_path}: {e}")
            return None
    
    def _get_face_embedding(self, face_file: Path) -> Optional[np.ndarray]:
        """Get face embedding from file."""
        try:
            img = cv2.imread(str(face_file))
            if img is None:
                return None
            return self.detect_manager.get_embedding(img)
        except Exception as e:
            self.logger.error(f"Error getting embedding for {face_file}: {e}")
            return None
    
    def _estimate_folder_quality(self, folder_path: Path) -> float:
        """Estimate average quality of faces in folder."""
        face_files = list(folder_path.glob("*.jpg"))[:10]  # Sample first 10
        if not face_files:
            return 0.0
        
        total_quality = 0
        valid_scores = 0
        
        for face_file in face_files:
            score = self._calculate_face_quality_score(face_file)
            if score > 0:
                total_quality += score
                valid_scores += 1
        
        avg_quality = total_quality / valid_scores if valid_scores > 0 else 0.0
        self.logger.debug(f"Average quality for {folder_path.name}: {avg_quality:.3f} "
                        f"(from {valid_scores}/{len(face_files)} faces)")
        
        return avg_quality
    
    def _calculate_adaptive_threshold(self, folder_size: int, avg_quality: float) -> float:
        """Calculate adaptive threshold based on folder characteristics."""
        base_threshold = self.similarity_threshold
        
        # Larger folders can have lower threshold (more confident)
        size_factor = max(0.9, 1.0 - (folder_size - 10) * 0.01)
        
        # Higher quality folders can have lower threshold
        quality_factor = max(0.9, 2.0 - avg_quality)
        
        adaptive_threshold = base_threshold * size_factor * quality_factor
        adaptive_threshold = max(0.6, min(0.9, adaptive_threshold))  # Clamp between 0.6 and 0.9
        
        self.logger.debug(f"Adaptive threshold calculation: base={base_threshold:.3f}, "
                        f"size_factor={size_factor:.3f}, quality_factor={quality_factor:.3f}, "
                        f"final={adaptive_threshold:.3f}")
        
        return adaptive_threshold
    
    def _find_merges_adaptive(self, folder_representatives: Dict, 
                            folder_info: Dict) -> Dict[str, str]:
        """Find merges using adaptive thresholds."""
        self.logger.info("Finding merges with adaptive thresholds")
        
        merge_mapping = {}
        folder_names = sorted(folder_representatives.keys())
        merged_folders = set()
        
        for i, current_folder in enumerate(folder_names):
            if current_folder in merged_folders:
                continue
            
            current_faces = folder_representatives[current_folder]
            current_threshold = folder_info[current_folder]['threshold']
            
            for j in range(i + 1, len(folder_names)):
                compare_folder = folder_names[j]
                if compare_folder in merged_folders:
                    continue
                
                compare_faces = folder_representatives[compare_folder]
                compare_threshold = folder_info[compare_folder]['threshold']
                
                # Use the more conservative (higher) threshold
                threshold = max(current_threshold, compare_threshold)
                
                # Calculate max similarity
                max_similarity = 0
                valid_comparisons = 0
                for curr_face in current_faces:
                    for comp_face in compare_faces:
                        sim = self.detect_manager.compute_similarity(str(curr_face[0]), str(comp_face[0]))
                        self.stats['total_comparisons'] += 1
                        if sim is not None:
                            max_similarity = max(max_similarity, sim)
                            valid_comparisons += 1
                        else:
                            self.stats['failed_comparisons'] += 1
                
                self.logger.debug(f"{current_folder} vs {compare_folder}: "
                               f"threshold={threshold:.3f}, max_sim={max_similarity:.3f}, "
                               f"valid_comparisons={valid_comparisons}")
                
                if max_similarity >= threshold:
                    self.logger.info(f"ADAPTIVE MERGE: {compare_folder} -> {current_folder} "
                                   f"(threshold: {threshold:.3f}, similarity: {max_similarity:.3f})")
                    merge_mapping[compare_folder] = current_folder
                    merged_folders.add(compare_folder)
                    self.stats['successful_merges'] += 1
        
        return merge_mapping
    
    def _calculate_optimal_eps(self, distances: np.ndarray) -> float:
        """Calculate optimal eps parameter for DBSCAN based on distance distribution."""
        # Use the distance to the k-th nearest neighbor approach
        k = 4  # Common choice
        k_distances = []
        
        for i in range(len(distances)):
            row_distances = distances[i]
            row_distances_sorted = np.sort(row_distances)
            if len(row_distances_sorted) > k:
                k_distances.append(row_distances_sorted[k])
        
        k_distances.sort()
        
        # Use knee point detection or percentile-based approach
        eps = np.percentile(k_distances, 95)  # Use 95th percentile
        
        self.logger.info(f"Calculated optimal eps: {eps:.4f} (from {len(k_distances)} k-distances)")
        return eps
    
    def _create_new_groupings(self, output_dir: Path, face_to_folder: Dict, 
                            folder_faces: Dict, new_labels: np.ndarray):
        """Create new folder groupings based on clustering results."""
        self.logger.info("Creating new groupings based on clustering results")
        
        # Create mapping from new cluster labels to faces
        cluster_faces = defaultdict(list)
        for face_idx, cluster_label in enumerate(new_labels):
            if cluster_label != -1:  # Ignore noise points
                original_folder = face_to_folder[face_idx]
                # Find the actual face file for this index
                for folder_name, faces_list in folder_faces.items():
                    if folder_name == original_folder:
                        for face_file, file_idx in faces_list:
                            if file_idx == face_idx:
                                cluster_faces[cluster_label].append(face_file)
                                break
        
        # Create new folders based on clusters
        temp_dir = output_dir / "temp_reclustered"
        temp_dir.mkdir(exist_ok=True)
        
        for cluster_id, faces in cluster_faces.items():
            cluster_folder = temp_dir / f"person_{cluster_id:03d}"
            cluster_folder.mkdir(exist_ok=True)
            
            self.logger.info(f"Creating cluster {cluster_id} with {len(faces)} faces")
            
            for face_file in faces:
                dest_file = cluster_folder / face_file.name
                counter = 1
                while dest_file.exists():
                    stem = face_file.stem
                    suffix = face_file.suffix
                    dest_file = cluster_folder / f"{stem}_{counter:03d}{suffix}"
                    counter += 1
                shutil.copy2(face_file, dest_file)
        
        # Replace original folders with new clusters
        self.logger.info("Replacing original folders with new clusters")
        
        # Remove original person folders
        for folder in output_dir.glob("person_*"):
            if folder.is_dir() and folder != temp_dir:
                shutil.rmtree(folder)
        
        # Move new clusters to main directory
        for cluster_folder in temp_dir.glob("person_*"):
            dest_folder = output_dir / cluster_folder.name
            shutil.move(str(cluster_folder), str(dest_folder))
        
        # Remove temp directory
        temp_dir.rmdir()
        
        self.logger.info(f"Created {len(cluster_faces)} new person folders")
    
    def _extract_temporal_info(self, folder_path: Path) -> Dict:
        """Extract temporal information from face filenames."""
        import re
        
        temporal_data = {'timestamps': [], 'frame_numbers': []}
        
        # Pattern to match timestamps and frame numbers in filenames
        # Adjust this pattern based on your filename format
        timestamp_pattern = r'(\d+(?:\.\d+)?s)'
        frame_pattern = r'frame_(\d+)'
        
        for face_file in folder_path.glob("*.jpg"):
            filename = face_file.name
            
            # Extract timestamp
            timestamp_match = re.search(timestamp_pattern, filename)
            if timestamp_match:
                try:
                    timestamp = float(timestamp_match.group(1).replace('s', ''))
                    temporal_data['timestamps'].append(timestamp)
                except ValueError as e:
                    self.logger.warning(f"Could not parse timestamp from {filename}: {e}")
            
            # Extract frame number
            frame_match = re.search(frame_pattern, filename)
            if frame_match:
                try:
                    frame_num = int(frame_match.group(1))
                    temporal_data['frame_numbers'].append(frame_num)
                except ValueError as e:
                    self.logger.warning(f"Could not parse frame number from {filename}: {e}")
        
        self.logger.debug(f"Extracted temporal info from {folder_path.name}: "
                        f"{len(temporal_data['timestamps'])} timestamps, "
                        f"{len(temporal_data['frame_numbers'])} frame numbers")
        
        return temporal_data
    
    def _find_merges_with_temporal_weighting(self, folder_temporal_info: Dict, 
                                           output_dir: Path) -> Dict[str, str]:
        """Find merges considering temporal overlap."""
        self.logger.info("Finding merges with temporal weighting")
        
        merge_mapping = {}
        folder_names = sorted(folder_temporal_info.keys())
        merged_folders = set()
        
        for i, current_folder in enumerate(folder_names):
            if current_folder in merged_folders:
                continue
            
            current_timestamps = set(folder_temporal_info[current_folder]['timestamps'])
            
            for j in range(i + 1, len(folder_names)):
                compare_folder = folder_names[j]
                if compare_folder in merged_folders:
                    continue
                
                compare_timestamps = set(folder_temporal_info[compare_folder]['timestamps'])
                
                # Calculate temporal overlap
                overlap = len(current_timestamps & compare_timestamps)
                total_unique = len(current_timestamps | compare_timestamps)
                temporal_similarity = overlap / total_unique if total_unique > 0 else 0
                
                self.logger.debug(f"{current_folder} vs {compare_folder}: "
                               f"temporal_overlap={temporal_similarity:.3f}")
                
                # If high temporal overlap, use lower similarity threshold
                if temporal_similarity > 0.3:  # Significant temporal overlap
                    adjusted_threshold = self.similarity_threshold * 0.9
                    
                    # Calculate face similarity
                    face_sim = self._calculate_folder_similarity(
                        output_dir / current_folder, output_dir / compare_folder
                    )
                    
                    self.logger.debug(f"Temporal overlap detected: face_sim={face_sim:.3f}, "
                                   f"adjusted_threshold={adjusted_threshold:.3f}")
                    
                    if face_sim >= adjusted_threshold:
                        self.logger.info(f"TEMPORAL MERGE: {compare_folder} -> {current_folder} "
                                       f"(temporal_sim: {temporal_similarity:.3f}, "
                                       f"face_sim: {face_sim:.3f})")
                        merge_mapping[compare_folder] = current_folder
                        merged_folders.add(compare_folder)
                        self.stats['successful_merges'] += 1
        
        return merge_mapping
    
    def _calculate_folder_similarity(self, folder1: Path, folder2: Path) -> float:
        """Calculate similarity between two folders using representative faces."""
        rep1 = self._get_highest_quality_face(folder1)
        rep2 = self._get_highest_quality_face(folder2)
        
        if not rep1 or not rep2:
            self.logger.warning(f"Could not get representatives for {folder1.name} or {folder2.name}")
            return 0.0
        
        similarity = self.detect_manager.compute_similarity(str(rep1[0]), str(rep2[0]))
        return similarity if similarity is not None else 0.0
    
    def _execute_merges(self, output_dir: Path, merge_mapping: Dict[str, str]):
        """Execute folder merges."""
        if not merge_mapping:
            self.logger.info("No merges to execute")
            return
        
        self.logger.info(f"Executing {len(merge_mapping)} merges")
        
        for source_folder, target_folder in merge_mapping.items():
            source_path = output_dir / source_folder
            target_path = output_dir / target_folder
            
            if not source_path.exists():
                self.logger.warning(f"Source folder does not exist: {source_path}")
                continue
                
            if not target_path.exists():
                self.logger.warning(f"Target folder does not exist: {target_path}")
                continue
            
            self.logger.info(f"Merging {source_folder} -> {target_folder}")
            
            # Count files before merge
            source_files = list(source_path.glob("*.jpg"))
            target_files_before = len(list(target_path.glob("*.jpg")))
            
            # Move all files
            moved_files = 0
            for item in source_files:
                if item.is_file():
                    target_file = target_path / item.name
                    counter = 1
                    while target_file.exists():
                        stem = item.stem
                        suffix = item.suffix
                        target_file = target_path / f"{stem}_{counter:03d}{suffix}"
                        counter += 1
                    
                    try:
                        shutil.move(str(item), str(target_file))
                        moved_files += 1
                    except Exception as e:
                        self.logger.error(f"Error moving {item} to {target_file}: {e}")
            
            # Remove empty source folder
            try:
                remaining_files = list(source_path.iterdir())
                if not remaining_files:
                    source_path.rmdir()
                    self.logger.info(f"Removed empty folder: {source_folder}")
                else:
                    self.logger.warning(f"Folder {source_folder} not empty after merge: "
                                      f"{len(remaining_files)} items remaining")
            except Exception as e:
                self.logger.error(f"Error removing folder {source_folder}: {e}")
            
            target_files_after = len(list(target_path.glob("*.jpg")))
            self.logger.info(f"Merge complete: moved {moved_files} files, "
                           f"{target_folder} now has {target_files_after} files "
                           f"(was {target_files_before})")
        
        # Renumber folders
        self.logger.info("Renumbering folders")
        self._renumber_folders(output_dir)
    
    def _renumber_folders(self, output_dir: Path):
        """Renumber person folders sequentially."""
        person_folders = sorted([f for f in output_dir.iterdir() 
                               if f.is_dir() and f.name.startswith("person_")])
        
        self.logger.info(f"Renumbering {len(person_folders)} folders")
        
        # Create a temporary mapping to avoid conflicts
        temp_mapping = {}
        for i, folder in enumerate(person_folders):
            temp_name = f"temp_person_{i:03d}"
            temp_path = output_dir / temp_name
            
            try:
                folder.rename(temp_path)
                temp_mapping[temp_path] = f"person_{i:03d}"
            except Exception as e:
                self.logger.error(f"Error renaming {folder} to {temp_path}: {e}")
        
        # Rename from temp names to final names
        for temp_path, final_name in temp_mapping.items():
            final_path = output_dir / final_name
            try:
                temp_path.rename(final_path)
                self.logger.debug(f"Renamed {temp_path.name} to {final_name}")
            except Exception as e:
                self.logger.error(f"Error renaming {temp_path} to {final_path}: {e}")
        
        final_folders = sorted([f for f in output_dir.iterdir() 
                              if f.is_dir() and f.name.startswith("person_")])
        self.logger.info(f"Renumbering complete: {len(final_folders)} folders")
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return self.stats.copy()


def enhanced_regroup_person_folders(output_dir, detect_manager, method="multiple_representatives", 
                                   log_file=None, **kwargs):
    """
    Enhanced regrouping with multiple strategies and comprehensive logging.
    
    Args:
        output_dir: Directory containing person folders
        detect_manager: Face detection/embedding manager
        method: Regrouping method to use
        log_file: Optional log file path
        **kwargs: Additional parameters for specific methods
    """
    # Setup logging
    log_file_path = None
    if log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    regrouper = ImprovedRegrouper(detect_manager, similarity_threshold=0.75, log_file=log_file_path)
    output_path = Path(output_dir)
    
    if not output_path.exists():
        regrouper.logger.error(f"Output directory does not exist: {output_path}")
        return
    
    regrouper.logger.info(f"Starting enhanced regrouping with method: {method}")
    regrouper.logger.info(f"Working directory: {output_path.absolute()}")
    
    start_time = time.time()
    
    try:
        if method == "multiple_representatives":
            num_representatives = kwargs.get('num_representatives', 3)
            consensus_threshold = kwargs.get('consensus_threshold', 0.6)
            regrouper.regroup_with_multiple_representatives(
                output_path, 
                num_representatives=num_representatives, 
                consensus_threshold=consensus_threshold
            )
        elif method == "quality_based":
            regrouper.regroup_with_quality_scoring(output_path)
        elif method == "adaptive_threshold":
            regrouper.regroup_with_adaptive_threshold(output_path)
        elif method == "embedding_clustering":
            regrouper.regroup_with_embedding_clustering(output_path)
        elif method == "temporal_context":
            regrouper.regroup_with_temporal_context(output_path)
        else:
            raise ValueError(f"Unknown regrouping method: {method}")
        
        total_time = time.time() - start_time
        stats = regrouper.get_statistics()
        
        regrouper.logger.info("=" * 80)
        regrouper.logger.info("FINAL STATISTICS")
        regrouper.logger.info("=" * 80)
        regrouper.logger.info(f"Method: {method}")
        regrouper.logger.info(f"Total processing time: {total_time:.2f} seconds")
        regrouper.logger.info(f"Total comparisons: {stats['total_comparisons']}")
        regrouper.logger.info(f"Successful merges: {stats['successful_merges']}")
        regrouper.logger.info(f"Failed comparisons: {stats['failed_comparisons']}")
        if stats['total_comparisons'] > 0:
            success_rate = (stats['total_comparisons'] - stats['failed_comparisons']) / stats['total_comparisons']
            regrouper.logger.info(f"Comparison success rate: {success_rate:.1%}")
        regrouper.logger.info("=" * 80)
        
        return stats
        
    except Exception as e:
        regrouper.logger.error(f"Error during regrouping: {e}", exc_info=True)
        raise


# Example usage
if __name__ == "__main__":
    # This would typically be imported from your face detection module
    from dinov3_manager import FaceDINOManager
    
    # Example usage with different methods and logging
    output_directory = "output/hdbscan"
    log_directory = "logs"
    
    # Create log directory
    Path(log_directory).mkdir(exist_ok=True)
    
    # Method 1: Multiple representatives with consensus
    stats1 = enhanced_regroup_person_folders(
        output_directory, 
        detect_manager=FaceDINOManager(),  # Replace with your FaceDINOManager()
        method="multiple_representatives",
        log_file=f"{log_directory}/regrouping_multiple_reps.log",
        num_representatives=1,
        consensus_threshold=0.9
    )
    
    # Method 2: Quality-based regrouping
    # stats2 = enhanced_regroup_person_folders(
    #     output_directory,
    #     detect_manager=None,  # Replace with your FaceDINOManager()
    #     method="quality_based",
    #     log_file=f"{log_directory}/regrouping_quality.log"
    # )
    
    # Method 3: Adaptive threshold
    # stats3 = enhanced_regroup_person_folders(
    #     output_directory,
    #     detect_manager=None,  # Replace with your FaceDINOManager()
    #     method="adaptive_threshold",
    #     log_file=f"{log_directory}/regrouping_adaptive.log"
    # )
    
    print("Regrouping completed. Check log files for detailed information.")
