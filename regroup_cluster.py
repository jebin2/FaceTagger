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
from tqdm import tqdm

THRESHOLD = 0.9

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
    
    def __init__(self, detect_manager, similarity_threshold=THRESHOLD, log_file=None):
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
    
    def regroup_with_multiple_representatives(self, output_dir: Path, num_representatives=5, consensus_threshold=THRESHOLD):
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
            tqdm.write(f"Processing folder {i+1}/{len(person_folders)}: {folder.name}", end="\r")
            representatives = self._get_multiple_representatives(folder, num_representatives)
            if representatives:
                folder_representatives[folder.name] = representatives
                tqdm.write(f"Found {len(representatives)} representatives in {folder.name}", end="\r")
            else:
                tqdm.write(f"No valid representatives found in {folder.name}", end="\r")
        
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

        self.check_unknown(output_dir, folder_representatives, consensus_threshold)
        
        processing_time = time.time() - start_time
        self.stats['processing_time'] = processing_time
        self.logger.info("=" * 60)
        self.logger.info("REGROUPING COMPLETED")
        self.logger.info(f"Processing time: {processing_time:.2f} seconds")
        self.logger.info(f"Successful merges: {self.stats['successful_merges']}")
        self.logger.info(f"Total comparisons: {self.stats['total_comparisons']}")
        self.logger.info(f"Failed comparisons: {self.stats['failed_comparisons']}")
        self.logger.info("=" * 60)

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
    
    def _get_multiple_representatives(self, folder_path: Path, num_representatives: int) -> List[Path]:
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
        # representatives = [fs for fs in face_scores[:num_representatives]]
        representatives = []
        for fs in face_scores[:num_representatives]:
            self.detect_manager.get_normalized_embedding_from_cache(str(fs[0]))
            representatives.append(fs)
        
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
    
    def _find_merges_with_consensus(self, folder_representatives: Dict, consensus_threshold: float) -> Dict[str, str]:
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

                tqdm.write(f"Processing pair {current_pair}/{total_pairs}: "f"{current_folder} vs {compare_folder}", end="\r")
                
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

    def check_unknown(self, output_dir: Path, folder_representatives: Dict, consensus_threshold: float) -> Dict[str, str]:
        """
        Check unknown folders and move faces to the most similar person folders.
        
        Args:
            output_dir: Directory containing person and unknown folders
            folder_representatives: Dictionary mapping folder names to their representative faces
            consensus_threshold: Minimum similarity threshold for moving faces
        
        Returns:
            Dictionary mapping moved files to their destination folders
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING UNKNOWN FOLDER PROCESSING")
        self.logger.info("=" * 60)
        
        # Find all unknown folders
        unknown_folders = sorted([f for f in output_dir.iterdir() 
                                if f.is_dir() and f.name.startswith("unknown")])
        
        if not unknown_folders:
            self.logger.info("No unknown folders found")
            return {}
        
        self.logger.info(f"Found {len(unknown_folders)} unknown folders")
        
        if not folder_representatives:
            self.logger.warning("No folder representatives provided - cannot process unknown faces")
            return {}
        
        moved_faces = {}
        total_unknown_faces = 0
        total_moved_faces = 0
        
        # Process each unknown folder
        for unknown_folder in unknown_folders:
            self.logger.info(f"Processing unknown folder: {unknown_folder.name}")
            
            # Get all face files from unknown folder
            face_files = list(unknown_folder.glob("*.jpg")) + list(unknown_folder.glob("*.jpeg")) + list(unknown_folder.glob("*.png"))
            
            if not face_files:
                self.logger.warning(f"No face files found in {unknown_folder.name}")
                continue
            
            self.logger.info(f"Found {len(face_files)} faces in {unknown_folder.name}")
            total_unknown_faces += len(face_files)
            for face_file in tqdm(face_files, desc=f"Processing embedding"):
                self.detect_manager.get_normalized_embedding_from_cache(str(face_file))
            
            # Process each face file
            for face_file in tqdm(face_files, desc=f"Checking similarities"):
                best_match_folder = None
                best_similarity = 0.0
                best_match_details = {}
                
                # Compare against all person folder representatives
                for folder_name, representatives in folder_representatives.items():
                    if not representatives:
                        continue
                    
                    # Calculate similarities with all representatives in this folder
                    similarities = []
                    valid_comparisons = 0
                    
                    for representative_info in representatives:
                        # representative_info is a tuple: (face_file_path, quality_score)
                        representative_path = representative_info[0]
                        
                        # Calculate similarity between unknown face and representative
                        similarity = self.detect_manager.compute_similarity(
                            str(face_file), str(representative_path)
                        )
                        
                        self.stats['total_comparisons'] += 1
                        
                        if similarity is not None:
                            similarities.append(similarity)
                            valid_comparisons += 1
                        else:
                            self.stats['failed_comparisons'] += 1
                    
                    if not similarities:
                        self.logger.debug(f"No valid similarities calculated for {folder_name}")
                        continue
                    
                    # Use maximum similarity for this folder
                    max_similarity = max(similarities)
                    avg_similarity = sum(similarities) / len(similarities)
                    
                    self.logger.debug(f"Face {face_file.name} vs {folder_name}: "
                                f"max_sim={max_similarity:.3f}, avg_sim={avg_similarity:.3f}, "
                                f"valid_comparisons={valid_comparisons}")
                    
                    # Update best match if this folder has higher similarity
                    if max_similarity > best_similarity:
                        best_similarity = max_similarity
                        best_match_folder = folder_name
                        best_match_details = {
                            'max_similarity': max_similarity,
                            'avg_similarity': avg_similarity,
                            'valid_comparisons': valid_comparisons,
                            'total_representatives': len(representatives)
                        }
                
                # Move face if similarity exceeds threshold
                if best_match_folder and best_similarity >= consensus_threshold:
                    self.logger.info(f"MATCH FOUND: {face_file.name} -> {best_match_folder} "
                                f"(similarity: {best_similarity:.3f})")
                    
                    # Move the face to the best matching person folder
                    destination_folder = output_dir / best_match_folder
                    if not destination_folder.exists():
                        self.logger.error(f"Destination folder does not exist: {destination_folder}")
                        continue
                    
                    # Generate unique filename in destination
                    dest_file = destination_folder / face_file.name
                    counter = 1
                    while dest_file.exists():
                        stem = face_file.stem
                        suffix = face_file.suffix
                        dest_file = destination_folder / f"{stem}_unknown_{counter:03d}{suffix}"
                        counter += 1
                    
                    try:
                        # Move the file
                        shutil.move(str(face_file), str(dest_file))
                        moved_faces[str(face_file)] = best_match_folder
                        total_moved_faces += 1
                        
                        self.logger.info(f"Moved {face_file.name} to {best_match_folder} as {dest_file.name}")
                        self.logger.debug(f"Match details: {best_match_details}")
                        
                    except Exception as e:
                        self.logger.error(f"Error moving {face_file} to {dest_file}: {e}")
                
                else:
                    if best_match_folder:
                        self.logger.debug(f"No match for {face_file.name}: best similarity {best_similarity:.3f} "
                                        f"< threshold {consensus_threshold:.3f} (best folder: {best_match_folder})")
                    else:
                        self.logger.debug(f"No valid matches found for {face_file.name}")
            
            # Remove unknown folder if it's now empty
            remaining_files = list(unknown_folder.glob("*"))
            if not remaining_files:
                try:
                    unknown_folder.rmdir()
                    self.logger.info(f"Removed empty unknown folder: {unknown_folder.name}")
                except Exception as e:
                    self.logger.error(f"Error removing empty folder {unknown_folder.name}: {e}")
            else:
                self.logger.info(f"Unknown folder {unknown_folder.name} still contains {len(remaining_files)} files")
        
        # Summary statistics
        self.logger.info("=" * 60)
        self.logger.info("UNKNOWN FOLDER PROCESSING COMPLETED")
        self.logger.info(f"Total unknown faces processed: {total_unknown_faces}")
        self.logger.info(f"Total faces moved to person folders: {total_moved_faces}")
        self.logger.info(f"Faces remaining in unknown folders: {total_unknown_faces - total_moved_faces}")
        
        if total_unknown_faces > 0:
            move_rate = (total_moved_faces / total_unknown_faces) * 100
            self.logger.info(f"Move rate: {move_rate:.1f}%")
        
        # Log destination summary
        destination_summary = {}
        for source_file, dest_folder in moved_faces.items():
            destination_summary[dest_folder] = destination_summary.get(dest_folder, 0) + 1
        
        if destination_summary:
            self.logger.info("Destination folder summary:")
            for folder, count in sorted(destination_summary.items()):
                self.logger.info(f"  {folder}: {count} faces")
        
        self.logger.info("=" * 60)
        
        return moved_faces
    
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
    
    def _find_merges_with_temporal_weighting(self, folder_temporal_info: Dict, output_dir: Path) -> Dict[str, str]:
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

    def move_folder_contents_recursively(self, source_path: Path, target_path: Path):
        """
        Recursively move contents from source to target, maintaining folder structure.
        """
        moved_files = 0
        
        def move_recursive(src_dir: Path, tgt_dir: Path, depth: int = 0):
            nonlocal moved_files
            indent = "  " * depth
            
            # Ensure target directory exists
            if not tgt_dir.exists():
                try:
                    tgt_dir.mkdir(parents=True)
                    self.logger.info(f"{indent}Created directory: {tgt_dir}")
                except Exception as e:
                    self.logger.error(f"{indent}Error creating directory {tgt_dir}: {e}")
                    return
            
            # Move all items in current directory
            for item in src_dir.iterdir():
                if item.is_file():
                    # Handle file
                    target_file = tgt_dir / item.name
                    counter = 1
                    while target_file.exists():
                        stem = item.stem
                        suffix = item.suffix
                        target_file = tgt_dir / f"{stem}_{counter:03d}{suffix}"
                        counter += 1
                    
                    try:
                        shutil.move(str(item), str(target_file))
                        moved_files += 1
                        self.logger.debug(f"{indent}Moved file: {item.name} -> {target_file}")
                    except Exception as e:
                        self.logger.error(f"{indent}Error moving {item} to {target_file}: {e}")
                
                elif item.is_dir():
                    # Handle subdirectory recursively
                    target_subdir = tgt_dir / item.name
                    self.logger.info(f"{indent}Processing subdirectory: {item.name}")
                    move_recursive(item, target_subdir, depth + 1)
                    
                    # Remove empty source subdirectory
                    try:
                        if not any(item.iterdir()):
                            item.rmdir()
                            self.logger.info(f"{indent}Removed empty directory: {item}")
                        else:
                            self.logger.warning(f"{indent}Directory not empty after move: {item}")
                    except Exception as e:
                        self.logger.error(f"{indent}Error removing directory {item}: {e}")
        
        # Start recursive move
        move_recursive(source_path, target_path)
        return moved_files

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
            target_files_before = len(list(target_path.glob("*.jpg")))
            moved_files = self.move_folder_contents_recursively(source_path, target_path)

            # Remove empty source folder
            try:
                # remaining_files = list(source_path.iterdir())
                remaining_files = [f for f in source_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
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


def enhanced_regroup_person_folders(output_dir, detect_manager, method="multiple_representatives", log_file=None, **kwargs):
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
    
    regrouper = ImprovedRegrouper(detect_manager, similarity_threshold=THRESHOLD, log_file=log_file_path)
    output_path = Path(output_dir)
    
    if not output_path.exists():
        regrouper.logger.error(f"Output directory does not exist: {output_path}")
        return
    
    regrouper.logger.info(f"Starting enhanced regrouping with method: {method}")
    regrouper.logger.info(f"Working directory: {output_path.absolute()}")
    
    start_time = time.time()
    
    try:
        if method == "multiple_representatives":
            num_representatives = kwargs.get('num_representatives', 1)
            consensus_threshold = kwargs.get('consensus_threshold', THRESHOLD)
            regrouper.regroup_with_multiple_representatives(
                output_path, 
                num_representatives=num_representatives, 
                consensus_threshold=consensus_threshold
            )
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
        consensus_threshold=THRESHOLD
    )
    
    # Method 2: Quality-based regrouping
    stats2 = enhanced_regroup_person_folders(
        output_directory,
        detect_manager=None,  # Replace with your FaceDINOManager()
        method="temporal_context",
        log_file=f"{log_directory}/regrouping_quality.log"
    )
    
    print("Regrouping completed. Check log files for detailed information.")
