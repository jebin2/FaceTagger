import time
import traceback
import cv2
import os
import numpy as np
from pathlib import Path
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import threading
from queue import Queue
import logging
from detect_manager import DetectManager
from tqdm import tqdm
import re
import pickle

# Configure logging for better progress tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXTRACT_FACES_FOLDER = "extracted_faces"

ALL_FRAMES_DIR = "all_frames_dir"

class VideoProcessor:
	def __init__(self, detect_manager: DetectManager, num_workers=None):
		self.detect_manager = detect_manager
		self.num_workers = num_workers or min(cpu_count(), 8)  # Cap at 8 to avoid memory issues
		
	def extract_faces_from_source(self, source_path, frame_skip=10):
		"""
		Simple face extraction:
		1. If video: extract all frames first, then process
		2. If folder: process existing frames
		3. For each frame: call detect(frame, frame_path) and save results
		"""
		source_path = Path(source_path)
		
		# Check for existing results
		faces_dir = Path(EXTRACT_FACES_FOLDER)
		metadata_file = faces_dir / 'face_metadata.pkl'
		
		if metadata_file.exists() and metadata_file.stat().st_size > 0:
			try:
				with open(metadata_file, 'rb') as f:
					return pickle.load(f)
			except (EOFError, pickle.UnpicklingError):
				logger.warning(f"Corrupted pickle file found at {metadata_file}, ignoring.")
		
		# Clean and create output directory
		if faces_dir.exists():
			shutil.rmtree(faces_dir)
		faces_dir.mkdir(parents=True, exist_ok=True)
		
		global ALL_FRAMES_DIR
		
		if source_path.is_file():
			# Step 1: Extract all frames from video first
			logger.info(f"Extracting frames from video: {source_path}")
			if os.path.exists(ALL_FRAMES_DIR):
				shutil.rmtree(ALL_FRAMES_DIR)
			os.makedirs(ALL_FRAMES_DIR)
			
			self._extract_all_frames(str(source_path), frame_skip)
			frames_folder = Path(ALL_FRAMES_DIR)
		elif source_path.is_dir():
			# Use existing frame folder
			frames_folder = source_path
		else:
			raise ValueError(f"Source path does not exist: {source_path}")
		
		# Step 2: Process all frames
		logger.info(f"Processing frames from: {frames_folder}")
		
		# Load model once
		self.detect_manager.load()
		
		try:
			all_face_data = self._process_frames_simple(frames_folder, faces_dir)
		finally:
			self.detect_manager.cleanup()
		
		# Save metadata
		with open(metadata_file, 'wb') as f:
			pickle.dump(all_face_data, f)
		
		logger.info(f"Extraction complete: {len(all_face_data)} faces saved")
		return all_face_data

	def _extract_all_frames(self, video_path, frame_skip):
		"""Extract all frames from video and save to ALL_FRAMES_DIR."""
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			logger.error(f"Could not open video: {video_path}")
			return
		
		fps = cap.get(cv2.CAP_PROP_FPS)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		
		frame_idx = 0
		saved_frames = 0
		
		logger.info(f"Extracting frames: total={total_frames}, fps={fps:.2f}, skip={frame_skip}")
		
		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				break
			
			if frame_idx % frame_skip == 0:
				timestamp = round(frame_idx / fps, 2)
				frame_filename = f'frame_{timestamp}_{frame_idx:05d}.jpg'
				frame_path = os.path.join(ALL_FRAMES_DIR, frame_filename)
				cv2.imwrite(frame_path, frame)
				saved_frames += 1
				
				if saved_frames % 100 == 0:
					print(f"\rExtracted {saved_frames} frames...", end='', flush=True)
			
			frame_idx += 1
		
		print(f"\nExtracted {saved_frames} frames to {ALL_FRAMES_DIR}")
		cap.release()

	def _process_frames_simple(self, frames_folder, faces_dir):
		"""Process all frames by calling detect() for each."""
		frames_folder = Path(frames_folder)
		
		# Get all image files
		image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
		image_files = []
		for ext in image_extensions:
			image_files.extend(frames_folder.glob(ext))
			image_files.extend(frames_folder.glob(ext.upper()))
		
		image_files = sorted(image_files, key=lambda x: x.name)
		
		if not image_files:
			logger.error(f"No image files found in: {frames_folder}")
			return []
		
		logger.info(f"Processing {len(image_files)} image files")
		
		all_face_data = []
		face_counter = 0
		
		for i, image_file in enumerate(image_files):
			try:
				# Load frame
				frame = cv2.imread(str(image_file))
				if frame is None:
					continue
				
				# Call detect with frame and file path
				detected_faces = self.detect_manager.detect(frame, str(image_file))
				
				# Save each detected face
				for face in detected_faces:
					face_data = self._save_detected_face(
						face, frame, faces_dir, face_counter, str(image_file), i
					)
					if face_data:
						all_face_data.append(face_data)
						face_counter += 1
				
				# Progress
				print(f"\rProcessed {i}/{len(image_files)} images, found {len(all_face_data)} faces", end='', flush=True)
					
			except Exception as e:
				logger.warning(f"Error processing {image_file}: {e}")
		
		print()
		return all_face_data

	def _save_detected_face(self, face, frame, faces_dir, face_id, source_file_path, frame_number):
		"""Save detected face exactly as returned from detector."""
		try:
			bbox = face['bbox']  # Keep exactly as detector returned
			embedding = face.get('embedding')
			confidence = face.get('confidence', face.get('score', 1.0))
			
			left, top, right, bottom = bbox
			
			# Basic validation
			if right <= left or bottom <= top:
				return None
			
			# Extract face from frame
			face_img = frame[top:bottom, left:right]
			
			if face_img.size == 0:
				return None
			
			# Save face image
			filename = Path(source_file_path).stem
			face_filename = f"{filename}_face_{face_id:05d}.jpg"
			face_path = faces_dir / face_filename
			
			cv2.imwrite(str(face_path), face_img)
			
			return {
				'face_id': face_id,
				'filename': face_filename,
				'frame_number': frame_number,
				'embedding': embedding,
				'bbox': bbox,  # Exact bbox from detector
				'confidence': confidence,
				'source_file_path': source_file_path
			}
			
		except Exception as e:
			logger.warning(f"Error saving face {face_id}: {e}")
			return None

	def annotate_video(self, video_path, faces_dir="output/hdbscan", output_path="annotated_video.mp4"):
		"""
		Annotate clustered faces back into the video with their cluster labels.

		Args:
			video_path: Path to the original video
			faces_dir: Folder containing HDBSCAN cluster subfolders (person_000, person_001, etc.)
			output_path: Path to save the annotated video
		"""
		# Regex to parse filenames like scene_0055_at_456.70s_face_00078_(83, 243, 151, 175).jpg
		filename_pattern = re.compile(
			r".*?_(\d+(?:\.\d+)?)_(\d+)_face_\d+_\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)\.(\w+)"
		)

		# Map: timestamp (seconds) -> list of (bbox, label)
		annotations = {}

		for person_folder in Path(faces_dir).glob("person_*"):
			label = person_folder.name
			for face_file in person_folder.glob("*.jpg"):
				match = filename_pattern.match(face_file.name)
				if not match:
					continue
				timestamp, frame_number, left, top, right, bottom, ext = match.groups()
				timestamp = float(timestamp)
				bbox = (int(left), int(top), int(right), int(bottom))
				annotations.setdefault(round(timestamp, 2), []).append((bbox, label))

		# Open video
		cap = cv2.VideoCapture(str(video_path))
		if not cap.isOpened():
			print(f"❌ Could not open video: {video_path}")
			return

		fps = cap.get(cv2.CAP_PROP_FPS)
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

		# Setup VideoWriter
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

		frame_idx = 0
		with tqdm(total=total_frames, desc="Annotating video", unit="frame") as pbar:
			while cap.isOpened():
				ret, frame = cap.read()
				if not ret:
					break

				# Get timestamp for this frame
				timestamp = round(frame_idx / fps, 2)
				if timestamp in annotations:
					for bbox, label in annotations[timestamp]:
						left, top, right, bottom = bbox
						cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
						cv2.putText(
							frame, label, (left, top - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
						)

				out.write(frame)
				frame_idx += 1
				pbar.update(1)

		cap.release()
		out.release()
		print(f"✅ Annotated clustered video saved to {output_path}")
	
	def _print_progress(self, frame_count, total_frames, fps, faces_found, start_time):
		"""Print optimized progress information"""
		elapsed = time.time() - start_time
		progress = frame_count / total_frames if total_frames > 0 else 0
		estimated_total = elapsed / progress if progress > 0 else 0
		remaining = estimated_total - elapsed
		
		elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
		remaining_min, remaining_sec = divmod(int(remaining), 60)
		
		current_sec = frame_count / fps if fps > 0 else 0
		current_min, current_s = divmod(int(current_sec), 60)
		
		fps_processed = frame_count / elapsed if elapsed > 0 else 0
		
		print(f"\rProgress: {progress:.1%} | Frame: {frame_count}/{total_frames} | "
			  f"Video: {current_min:02d}:{current_s:02d} | Faces: {faces_found} | "
			  f"Speed: {fps_processed:.1f} fps | Remaining: {remaining_min:02d}:{remaining_sec:02d}", 
			  end='', flush=True)


def save_grouped_faces_optimized(face_data, labels, method_name):
	"""Enhanced version with parallel file operations and original file copying"""
	output_dir = Path("output") / method_name
	
	if output_dir.exists():
		shutil.rmtree(output_dir)
	output_dir.mkdir(parents=True)

	def copy_face_and_original_file(args):
		face, label = args
		group_name = f"person_{label:03d}" if label != -1 else "unknown"
		group_path = output_dir / group_name
		group_path.mkdir(exist_ok=True)
		
		# Copy the extracted face file
		source_face_path = Path("extracted_faces") / face['filename']
		if source_face_path.exists():
			shutil.copy2(source_face_path, group_path)  # copy2 preserves metadata
		
		# Copy the original frame/image file if it exists
		if 'source_file_path' in face and face['source_file_path']:
			original_file_path = Path(face['source_file_path'])
			if original_file_path.exists():
				# Create a subdirectory for original files
				originals_dir = group_path / "original_frames"
				originals_dir.mkdir(exist_ok=True)
				
				# Create a unique name for the original file to avoid conflicts
				original_filename = f"{face.get('original_filename', original_file_path.name)}"
				if face.get('face_id') is not None:
					# Add face ID to make filename unique
					stem = Path(original_filename).stem
					ext = Path(original_filename).suffix
					original_filename = f"{stem}_face{face['face_id']:05d}{ext}"
				
				dest_original_path = originals_dir / original_filename
				
				# Avoid overwriting if file already exists
				counter = 1
				while dest_original_path.exists():
					stem = Path(original_filename).stem
					ext = Path(original_filename).suffix
					dest_original_path = originals_dir / f"{stem}_{counter:03d}{ext}"
					counter += 1
				
				try:
					shutil.copy2(original_file_path, dest_original_path)
				except Exception as e:
					logger.warning(f"Could not copy original file {original_file_path}: {e}")

	# Parallel file copying
	with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
		executor.map(copy_face_and_original_file, zip(face_data, labels))
	
	# Print summary
	label_counts = Counter(labels)
	num_persons = len(label_counts) - (1 if -1 in label_counts else 0)
	num_unknown = label_counts.get(-1, 0)
	
	logger.info(f"Results for {method_name}: {num_persons} persons, {num_unknown} unknown faces")
	logger.info(f"Saved to: {output_dir}")
	logger.info(f"Each group contains both extracted faces and original frame files")


def clustering_pipeline(faces_data, max_workers=None):
	"""Run clustering algorithms in parallel where possible"""
	from group_face import (cluster_with_hdbscan, cluster_with_spectral, 
						   cluster_with_adaptive_similarity, cluster_with_agglomerative, 
						   cluster_with_affinity_propagation, cluster_with_chinese_whispers, compare_and_group_faces_with_fr,
						   get_similarity)
	
	face_encodings = [face['embedding'] for face in faces_data if face['embedding'] is not None]
	
	# if not face_encodings:
	# 	logger.warning("No face embeddings found for clustering")
	# 	return
	
	# Convert to numpy array for better performance
	face_encodings = np.array(face_encodings)
	
	clustering_techniques = [
		("hdbscan", lambda: cluster_with_hdbscan(face_encodings)),
		# ("spectral_clustering", lambda: cluster_with_spectral(face_encodings)),
		# ("adaptive_similarity", lambda: cluster_with_adaptive_similarity(face_encodings)),
		# ("agglomerative", lambda: cluster_with_agglomerative(face_encodings)),
		# ("affinity_propagation", lambda: cluster_with_affinity_propagation(face_encodings)),
		# ("chinese_whispers", lambda: cluster_with_chinese_whispers(face_encodings)),
		# ("get_similarity", lambda: get_similarity(face_encodings)),
		# ("compare_and_group_faces_with_fr", lambda: compare_and_group_faces_with_fr(EXTRACT_FACES_FOLDER)),
	]

	# Some algorithms can run in parallel, others are already multi-threaded
	def run_clustering(name_func_pair):
		name, func = name_func_pair
		try:
			start_time = time.time()
			labels = func()
			elapsed = time.time() - start_time
			logger.info(f"{name} completed in {elapsed:.2f} seconds")
			save_grouped_faces_optimized(faces_data, labels, name)
			return name, labels, elapsed
		except Exception as e:
			logger.error(f"Error in {name} clustering: {e} {traceback.format_exc()}")
			return name, None, 0

	logger.info("Starting parallel clustering pipeline...")
	
	# Run clustering algorithms
	results = []
	with ThreadPoolExecutor(max_workers=max_workers or 3) as executor:  # Limit to avoid memory issues
		futures = [executor.submit(run_clustering, technique) for technique in clustering_techniques]
		for future in futures:
			results.append(future.result())
	
	# Print summary
	successful_runs = [r for r in results if r[1] is not None]
	logger.info(f"Completed {len(successful_runs)}/{len(clustering_techniques)} clustering algorithms")

	logger.info("Enhanced clustering pipeline completed")
	for name, _, elapsed in successful_runs:
		logger.info(f"  {name}: {elapsed:.2f}s")

def calculate_face_quality_score(face_file: Path) -> float:
	"""
	Calculate comprehensive quality score for a face.
	Considers: size, sharpness, contrast, face detection confidence.
	"""
	try:
		img = cv2.imread(str(face_file))
		if img is None:
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
		logger.warning(f"Error calculating quality for {face_file}: {e}")
		return 0.0

def regroup_person_folders(output_dir, detect_manager=None, similarity_threshold=0.7):
	"""
	Regroup person folders by merging similar faces.
	
	Args:
		output_dir: Directory containing person_XXX folders
		similarity_threshold: Minimum similarity to merge folders (default 0.7)
		sample_size: Number of sample faces to use for comparison per folder
	"""
	output_path = Path(output_dir)
	if not output_path.exists():
		logger.error(f"Output directory does not exist: {output_dir}")
		return
	
	# Get all person folders
	person_folders = sorted([f for f in output_path.iterdir() 
						   if f.is_dir() and f.name.startswith("person_")])
	
	if len(person_folders) < 2:
		logger.info("Less than 2 person folders found, no regrouping needed")
		return
	
	logger.info(f"Starting regrouping of {len(person_folders)} person folders")
	
	# Extract representative faces from each folder
	folder_representatives = {}
	for folder in person_folders:
		representatives = get_representative_face_with_largest_area(folder)
		if representatives:
			folder_representatives[folder.name] = representatives
	
	# Track which folders have been merged
	merged_folders = set()
	merge_mapping = {}  # source_folder -> target_folder
	
	# Compare folders sequentially
	folder_names = sorted(folder_representatives.keys())
	
	for i, current_folder in enumerate(folder_names):
		if current_folder in merged_folders:
			continue
			
		current_faces = folder_representatives[current_folder]
		
		# Compare with all subsequent folders
		for j in range(i + 1, len(folder_names)):
			compare_folder = folder_names[j]
			
			if compare_folder in merged_folders:
				continue
				
			compare_faces = folder_representatives[compare_folder]
			
			# Calculate similarity between folders
			max_similarity = detect_manager.compute_similarity(current_faces, compare_faces)
			
			if max_similarity >= similarity_threshold:
				logger.info(f"Merging {compare_folder} into {current_folder} (similarity: {max_similarity:.3f})")
				
				# Mark for merging
				merge_mapping[compare_folder] = current_folder
				merged_folders.add(compare_folder)
	
	# Execute the merges
	execute_folder_merges(output_path, merge_mapping)
	
	# Renumber remaining folders
	renumber_person_folders(output_path)
	
	remaining_folders = len([f for f in output_path.iterdir() 
						   if f.is_dir() and f.name.startswith("person_")])
	
	logger.info(f"Regrouping complete: {len(person_folders)} -> {remaining_folders} person folders")

def get_representative_face_with_largest_area(folder_path):
    """Get highest quality face instead of largest."""
    folder_path = Path(folder_path)
    face_files = list(folder_path.glob("*.jpg"))
    
    best_score = 0
    best_file = None
    
    for face_file in face_files:
        score = calculate_face_quality_score(face_file)  # From the artifact above
        if score > best_score:
            best_score = score
            best_file = face_file
    
    return best_file

def execute_folder_merges(output_path, merge_mapping):
	"""
	Execute the actual folder merges.
	
	Args:
		output_path: Base output directory
		merge_mapping: Dictionary of source_folder -> target_folder
	"""
	for source_folder, target_folder in merge_mapping.items():
		source_path = output_path / source_folder
		target_path = output_path / target_folder
		
		if not source_path.exists() or not target_path.exists():
			logger.warning(f"Skipping merge: {source_folder} -> {target_folder} (folder not found)")
			continue
		
		logger.info(f"Moving files from {source_folder} to {target_folder}")
		
		# Move all files from source to target
		for item in source_path.iterdir():
			if item.is_file():
				# Handle filename conflicts
				target_file = target_path / item.name
				counter = 1
				while target_file.exists():
					stem = item.stem
					suffix = item.suffix
					target_file = target_path / f"{stem}_{counter:03d}{suffix}"
					counter += 1
				
				shutil.move(str(item), str(target_file))
			elif item.is_dir():
				# Handle subdirectories (like original_frames)
				target_subdir = target_path / item.name
				if target_subdir.exists():
					# Move contents if subdirectory already exists
					for subitem in item.iterdir():
						target_subfile = target_subdir / subitem.name
						counter = 1
						while target_subfile.exists():
							stem = subitem.stem
							suffix = subitem.suffix
							target_subfile = target_subdir / f"{stem}_{counter:03d}{suffix}"
							counter += 1
						shutil.move(str(subitem), str(target_subfile))
					item.rmdir()
				else:
					shutil.move(str(item), str(target_subdir))
		
		# Remove empty source folder
		try:
			source_path.rmdir()
		except OSError:
			logger.warning(f"Could not remove {source_path} (not empty)")

def renumber_person_folders(output_path):
	"""
	Renumber person folders to be sequential (person_000, person_001, etc.)
	
	Args:
		output_path: Base output directory
	"""
	person_folders = sorted([f for f in output_path.iterdir() 
						   if f.is_dir() and f.name.startswith("person_")])
	
	for i, folder in enumerate(person_folders):
		new_name = f"person_{i:03d}"
		if folder.name != new_name:
			new_path = output_path / new_name
			# Handle conflicts
			while new_path.exists():
				i += 1
				new_name = f"person_{i:03d}"
				new_path = output_path / new_name
			
			logger.info(f"Renaming {folder.name} to {new_name}")
			folder.rename(new_path)


if __name__ == "__main__":
	detect_type = "moondream2"
	
	# Performance settings
	frame_skip = 10  # Increased for faster processing
	
	# --- Select Detector ---
	if detect_type == "insight_face":
		from insight_face_manager import InsightFaceManager
		detect_manager = InsightFaceManager()
	elif detect_type == "face_recognition":
		from face_recognition_manager import FaceRecognitionManager
		detect_manager = FaceRecognitionManager(is_anime=False)
	elif detect_type == "deepface":
		from deepface_manager import DeepFaceManager
		detect_manager = DeepFaceManager()
	elif detect_type == "dinov3":
		from dinov3_manager import FaceDINOManager
		detect_manager = FaceDINOManager()
	elif detect_type == "moondream2":
		from moondream2_face_manager import MoonDream2FaceManager
		detect_manager = MoonDream2FaceManager()
	# elif detect_type == "mediapipe": # worst
	# 	from mediapipe_face_manager import MediaPipeFaceManager
	# 	detect_manager = MediaPipeFaceManager(model_selection=1, min_detection_confidence=0.7)
	# elif detect_type == "yolo": # not good
	# 	num_workers = 1
	# 	from yolo_person_manager import YOLOPersonManager
	# 	detect_manager = YOLOPersonManager(conf_threshold=0.7)
	else:
		raise ValueError(f"Unknown detect_type: {detect_type}")

	# --- Run Optimized Pipeline ---
	processor = VideoProcessor(detect_manager, num_workers=1)
	
	start_total = time.time()
	faces_data = []

	faces_data = processor.extract_faces_from_source(
		"input.mp4",
		frame_skip=frame_skip
	)

	clustering_pipeline(faces_data)

	# Perform regrouping
	# logger.info("Starting post-clustering regrouping...")
	# method_name="hdbscan"
	# output_dir = f"output/{method_name}"
	# regroup_person_folders(output_dir, detect_manager=detect_manager, similarity_threshold=0.9)

	# Example usage with different methods and logging
	from regroup_cluster import enhanced_regroup_person_folders
	output_directory = "output/hdbscan"
	log_directory = "logs"

	# Create log directory
	Path(log_directory).mkdir(exist_ok=True)
	from dinov3_manager import FaceDINOManager
	# Method 1: Multiple representatives with consensus
	stats1 = enhanced_regroup_person_folders(
		output_directory, 
		detect_manager=FaceDINOManager(),  # Replace with your FaceDINOManager()
		method="multiple_representatives",
		log_file=f"{log_directory}/regrouping_multiple_reps.log",
		num_representatives=1,
	)

	# Method 2: Quality-based regrouping
	stats2 = enhanced_regroup_person_folders(
		output_directory,
		detect_manager=FaceDINOManager(),  # Replace with your FaceDINOManager()
		method="temporal_context",
		log_file=f"{log_directory}/regrouping_quality.log"
	)

	total_elapsed = time.time() - start_total
	logger.info(f"\nTotal pipeline completed in {total_elapsed:.2f} seconds")
	logger.info(f"Processed {len(faces_data)} faces")
	# processor.annotate_video(videoPath)