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

# Configure logging for better progress tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXTRACT_FACES_FOLDER = "extracted_faces"

ALL_FRAMES_DIR = "all_frames_dir"

class VideoProcessor:
	def __init__(self, detect_manager: DetectManager, num_workers=None):
		self.detect_manager = detect_manager
		self.num_workers = num_workers or min(cpu_count(), 8)  # Cap at 8 to avoid memory issues
		
	def extract_faces_from_source(self, source_path, frame_skip=10, batch_size=32, max_faces_per_frame=10, min_face_size=30, use_gpu_decode=True, quality_threshold=0.7):
		"""
		Extract faces from either a video file or a folder containing frames.
		
		Args:
			source_path: Path to video file or folder containing frame images
			frame_skip: Number of frames to skip between processing (for videos) or step size for frame folders
			batch_size: Number of frames to process in parallel
			max_faces_per_frame: Maximum faces to extract per frame
			min_face_size: Minimum face size in pixels
			use_gpu_decode: Use GPU acceleration for video decoding (ignored for frame folders)
			quality_threshold: Minimum confidence threshold for face detection
		"""
		source_path = Path(source_path)
		global ALL_FRAMES_DIR
		if source_path.is_file():
			# Process as video file
			logger.info(f"Processing video file: {source_path}")
			if os.path.exists(ALL_FRAMES_DIR):
				shutil.rmtree(ALL_FRAMES_DIR)
			return self.extract_faces_from_video_optimized(
				str(source_path), frame_skip, batch_size, max_faces_per_frame, 
				min_face_size, use_gpu_decode, quality_threshold
			)
		elif source_path.is_dir():
			# Process as frame folder
			logger.info(f"Processing frame folder: {source_path}")
			ALL_FRAMES_DIR = source_path
			return self.extract_faces_from_frames_folder(
				source_path, frame_skip, batch_size, max_faces_per_frame, 
				min_face_size, quality_threshold
			)
		else:
			raise ValueError(f"Source path does not exist or is neither a file nor directory: {source_path}")

	def extract_faces_from_frames_folder(self, frames_folder, frame_skip=10, batch_size=32, max_faces_per_frame=10, min_face_size=30, quality_threshold=0.7):
		"""
		Extract faces from a folder containing frame images.
		
		Args:
			frames_folder: Path to folder containing frame images
			frame_skip: Step size for processing frames (e.g., process every 10th frame)
			batch_size: Number of frames to process in parallel
			max_faces_per_frame: Maximum faces to extract per frame
			min_face_size: Minimum face size in pixels
			quality_threshold: Minimum confidence threshold for face detection
		"""
		if not isinstance(self.detect_manager, DetectManager):
			raise ValueError("detect_manager must be an instance of DetectManager")

		frames_folder = Path(frames_folder)
		if not frames_folder.exists():
			raise ValueError(f"Frames folder does not exist: {frames_folder}")

		# --- Directory Setup ---
		faces_dir = Path(EXTRACT_FACES_FOLDER)
		if faces_dir.exists():
			shutil.rmtree(faces_dir)
		faces_dir.mkdir(parents=True, exist_ok=True)
		
		logger.info(f"Starting face extraction from frames folder '{frames_folder}'")
		logger.info(f"Using {self.num_workers} workers, batch_size={batch_size}")

		# --- Get all image files ---
		image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
		image_files = []
		for ext in image_extensions:
			image_files.extend(frames_folder.glob(ext))
			image_files.extend(frames_folder.glob(ext.upper()))
		
		# Sort files naturally (frame_001.jpg, frame_002.jpg, etc.)
		image_files = sorted(image_files, key=lambda x: x.name)
		
		if not image_files:
			logger.error(f"No image files found in folder: {frames_folder}")
			return []
		
		# Apply frame skip
		selected_files = image_files[::frame_skip]
		total_frames = len(selected_files)
		
		logger.info(f"Found {len(image_files)} image files, processing {total_frames} with skip={frame_skip}")
		
		# --- Load model once ---
		self.detect_manager.load()
		
		# --- Process frames with optimizations ---
		all_face_data = self._process_frames_from_folder(
			selected_files, batch_size, faces_dir, 
			max_faces_per_frame, min_face_size, quality_threshold
		)
		
		self.detect_manager.cleanup()
		
		def convert_np(obj):
			if isinstance(obj, dict):
				return {k: convert_np(v) for k, v in obj.items()}
			elif isinstance(obj, list):
				return [convert_np(x) for x in obj]
			elif isinstance(obj, tuple):
				return tuple(convert_np(x) for x in obj)
			elif isinstance(obj, np.integer):
				return int(obj)
			elif isinstance(obj, np.floating):
				return float(obj)
			else:
				return obj

		data_clean = convert_np(all_face_data)

		logger.info(f"Extraction complete: {len(all_face_data)} faces saved to '{faces_dir}'")
		return all_face_data

	def _process_frames_from_folder(self, image_files, batch_size, faces_dir, max_faces_per_frame, min_face_size, quality_threshold):
		"""Process image files from folder in parallel batches"""
		all_face_data = []
		face_counter = 0
		start_time = time.time()
		total_files = len(image_files)
		
		# Process in batches
		for batch_start in range(0, total_files, batch_size):
			batch_end = min(batch_start + batch_size, total_files)
			batch_files = image_files[batch_start:batch_end]
			
			# Load frames for this batch
			frame_buffer = []
			frame_info_buffer = []
			
			for i, image_file in enumerate(batch_files):
				try:
					frame = cv2.imread(str(image_file))
					if frame is not None:
						# Handle different frame formats
						if len(frame.shape) == 2:  # Grayscale
							frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
						elif len(frame.shape) == 3 and frame.shape[2] == 4:  # BGRA
							frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
						
						processed_frame = self._preprocess_frame(frame)
						frame_buffer.append(processed_frame)
						frame_info_buffer.append({
							'original_frame': frame, 
							'frame_number': batch_start + i,
							'filename': image_file.name,
							'source_file_path': str(image_file)  # Add full path to source file
						})
				except Exception as e:
					logger.warning(f"Could not load image {image_file}: {e}")
			
			# Process this batch
			if frame_buffer:
				batch_faces = self._process_frame_batch(
					frame_buffer, frame_info_buffer, faces_dir, 
					face_counter, max_faces_per_frame, min_face_size, quality_threshold
				)
				all_face_data.extend(batch_faces)
				face_counter += len(batch_faces)
			
			# Progress update
			processed_frames = batch_end
			progress = processed_frames / total_files
			elapsed = time.time() - start_time
			
			if processed_frames % (batch_size * 2) == 0:  # Update every 2 batches
				fps_processed = processed_frames / elapsed if elapsed > 0 else 0
				estimated_total = elapsed / progress if progress > 0 else 0
				remaining = estimated_total - elapsed
				remaining_min, remaining_sec = divmod(int(remaining), 60)
				
				print(f"\rProgress: {progress:.1%} | Frames: {processed_frames}/{total_files} | "
					  f"Faces: {len(all_face_data)} | Speed: {fps_processed:.1f} fps | "
					  f"Remaining: {remaining_min:02d}:{remaining_sec:02d}", 
					  end='', flush=True)
		
		print()  # New line after progress
		return all_face_data
		
	def extract_faces_from_video_optimized(self, video_path, frame_skip=10, batch_size=32, max_faces_per_frame=10, min_face_size=30, use_gpu_decode=True, quality_threshold=0.7):
		"""
		Optimized face extraction with multiple performance improvements:
		- Batch processing for detection
		- Parallel frame processing
		- GPU-accelerated video decoding (if available)
		- Early quality filtering
		- Memory-efficient processing
		- Smart frame skipping based on motion detection
		"""
		if not isinstance(self.detect_manager, DetectManager):
			raise ValueError("detect_manager must be an instance of DetectManager")

		# --- Directory Setup ---
		faces_dir = Path(EXTRACT_FACES_FOLDER)
		if faces_dir.exists():
			shutil.rmtree(faces_dir)
		faces_dir.mkdir(parents=True, exist_ok=True)
		
		metadata_file = faces_dir / 'face_metadata.json'
		
		logger.info(f"Starting optimized face extraction from '{video_path}'")
		logger.info(f"Using {self.num_workers} workers, batch_size={batch_size}")

		# --- Video Setup with GPU acceleration ---
		cap = self._setup_video_capture(video_path, use_gpu_decode)
		if not cap.isOpened():
			logger.error(f"Could not open video file {video_path}")
			return []

		fps = cap.get(cv2.CAP_PROP_FPS)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		video_name = Path(video_path).stem  # Get video name without extension
		
		# --- Load model once ---
		self.detect_manager.load()
		
		# --- Process video with optimizations ---
		all_face_data = self._process_video_parallel(
			cap, fps, total_frames, frame_skip, batch_size, 
			faces_dir, max_faces_per_frame, min_face_size, quality_threshold, video_name
		)
		
		cap.release()
		self.detect_manager.cleanup()
		def convert_np(obj):
			if isinstance(obj, dict):
				return {k: convert_np(v) for k, v in obj.items()}
			elif isinstance(obj, list):
				return [convert_np(x) for x in obj]
			elif isinstance(obj, tuple):
				return tuple(convert_np(x) for x in obj)
			elif isinstance(obj, np.integer):
				return int(obj)
			elif isinstance(obj, np.floating):
				return float(obj)
			else:
				return obj

		data_clean = convert_np(all_face_data)
		# --- Save metadata ---
		# with open(metadata_file, 'w') as f:
		# 	json.dump(data_clean, f, indent=2)

		logger.info(f"Extraction complete: {len(all_face_data)} faces saved to '{faces_dir}'")
		return all_face_data
	
	def _setup_video_capture(self, video_path, use_gpu_decode):
		"""Setup video capture with proper format handling"""
		cap = cv2.VideoCapture(video_path)
		
		# Try to enable GPU decoding if requested and available
		if use_gpu_decode:
			try:
				# Attempt hardware acceleration (varies by system)
				cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
				# Force BGR color space to avoid format issues
				cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
			except:
				logger.warning("GPU decode not available, using CPU")
		
		# Ensure consistent color format
		cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
		
		return cap
	
	def _process_video_parallel(self, cap, fps, total_frames, frame_skip, batch_size, faces_dir, max_faces_per_frame, min_face_size, quality_threshold, video_name):
		"""Process video frames in parallel batches"""
		all_face_data = []
		face_counter = 0
		frame_count = 0
		start_time = time.time()
		
		# Frame buffer for batch processing
		frame_buffer = []
		frame_info_buffer = []
		
		# Motion detection for smart frame skipping
		prev_frame_gray = None
		motion_threshold = 1000  # Adjust based on video content
		if not os.path.exists(ALL_FRAMES_DIR):
			os.mkdir(ALL_FRAMES_DIR)

		frame_idx = 0
		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				break
			
			# Handle different frame formats that might come from video decoder
			if frame is None:
				frame_count += 1
				continue

			timestamp = round(frame_idx / fps, 2)
			frame_idx += 1
			# Ensure frame is in correct format (BGR)
			if len(frame.shape) == 2:  # Grayscale
				frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
			elif len(frame.shape) == 3 and frame.shape[2] == 1:  # Single channel
				frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
			elif len(frame.shape) == 3 and frame.shape[2] == 4:  # BGRA
				frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
				
			if frame_count % frame_skip == 0:
				frame_filename = f'frame_{timestamp}_{frame_count:05d}.jpg'
				cv2.imwrite(f'{ALL_FRAMES_DIR}/{frame_filename}', frame)
				
				# Smart frame skipping based on motion
				if self._should_process_frame(frame, prev_frame_gray, motion_threshold):
					# Resize frame for faster processing if it's too large
					processed_frame = self._preprocess_frame(frame)
					frame_buffer.append(processed_frame)
					frame_info_buffer.append({
						'original_frame': frame, 
						'frame_number': frame_count,
						'filename': frame_filename,
						'video_name': video_name,
						'source_file_path': f'{ALL_FRAMES_DIR}/{frame_filename}'
					})
					
					# Update motion detection reference - handle format safely
					if len(processed_frame.shape) == 3 and processed_frame.shape[2] == 3:
						prev_frame_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
					else:
						prev_frame_gray = processed_frame if len(processed_frame.shape) == 2 else processed_frame[:,:,0]
				
				# Process batch when full
				if len(frame_buffer) >= batch_size:
					batch_faces = self._process_frame_batch(
						frame_buffer, frame_info_buffer, faces_dir, 
						face_counter, max_faces_per_frame, min_face_size, quality_threshold
					)
					all_face_data.extend(batch_faces)
					face_counter += len(batch_faces)
					
					# Clear buffers
					frame_buffer.clear()
					frame_info_buffer.clear()
					
			# Progress update
			if frame_count % (frame_skip * 10) == 0:
				self._print_progress(frame_count, total_frames, fps, len(all_face_data), start_time)
				
			frame_count += 1
		
		# Process remaining frames in buffer
		if frame_buffer:
			batch_faces = self._process_frame_batch(
				frame_buffer, frame_info_buffer, faces_dir,
				face_counter, max_faces_per_frame, min_face_size, quality_threshold
			)
			all_face_data.extend(batch_faces)
			
		return all_face_data
	
	def _should_process_frame(self, frame, prev_frame_gray, motion_threshold):
		"""Determine if frame should be processed based on motion detection"""
		if prev_frame_gray is None:
			return True
		
		# Handle different color formats safely
		if len(frame.shape) == 3 and frame.shape[2] == 3:
			current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		elif len(frame.shape) == 3 and frame.shape[2] == 4:
			current_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
		elif len(frame.shape) == 2:
			current_gray = frame
		else:
			# Fallback: assume it's already grayscale or handle as-is
			current_gray = frame if len(frame.shape) == 2 else frame[:,:,0]
		
		if current_gray.shape != prev_frame_gray.shape:
			return True  # Process if shapes don't match
			
		diff = cv2.absdiff(prev_frame_gray, current_gray)
		motion_score = np.sum(diff)
		
		return motion_score > motion_threshold
	
	def _preprocess_frame(self, frame, max_width=1280):
		"""Preprocess frame for faster detection"""
		height, width = frame.shape[:2]
		
		# Resize if frame is too large
		if width > max_width:
			scale = max_width / width
			new_width = max_width
			new_height = int(height * scale)
			frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
			
		return frame
	
	def _process_frame_batch(self, frames, frame_infos, faces_dir, face_counter_start, max_faces_per_frame, min_face_size, quality_threshold):
		"""Process a batch of frames in parallel"""
		batch_results = []
		face_counter = face_counter_start
		
		# Use ThreadPoolExecutor for I/O bound operations (face detection)
		with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
			# Submit detection tasks
			future_to_frame = {
				executor.submit(self._detect_faces_in_frame, frame, frame_info, max_faces_per_frame): 
				(frame, frame_info) for frame, frame_info in zip(frames, frame_infos)
			}
			
			# Collect results
			for future in future_to_frame:
				try:
					detected_faces, frame_info = future.result(timeout=30)  # 30 second timeout
					
					# Process detected faces
					for face in detected_faces:
						if self._is_valid_face(face, min_face_size, quality_threshold):
							face_data = self._save_face_image(
								face, frame_info['original_frame'], faces_dir, 
								face_counter, frame_info
							)
							if face_data:
								batch_results.append(face_data)
								face_counter += 1
								
				except Exception as e:
					logger.error(f"Error processing frame batch: {e}")
					
		return batch_results
	
	def _detect_faces_in_frame(self, frame, frame_info, max_faces_per_frame):
		"""Detect faces in a single frame"""
		try:
			detected_faces = self.detect_manager.detect(frame)
			# Limit number of faces per frame to prevent memory issues
			if len(detected_faces) > max_faces_per_frame:
				# Sort by confidence if available, otherwise by face size
				detected_faces = sorted(detected_faces, 
									  key=lambda x: x.get('confidence', self._get_face_size(x['bbox'])), 
									  reverse=True)[:max_faces_per_frame]
			return detected_faces, frame_info
		except Exception as e:
			logger.error(f"Error detecting faces in frame: {e}")
			return [], frame_info
	
	def _is_valid_face(self, face, min_face_size, quality_threshold):
		"""Check if detected face meets quality requirements"""
		bbox = face['bbox']
		face_width = bbox[1] - bbox[3]  # right - left
		face_height = bbox[2] - bbox[0]  # bottom - top
		
		# Size check
		if min(face_width, face_height) < min_face_size:
			return False
			
		# Quality/confidence check if available
		confidence = face.get('confidence', 1.0)
		if confidence < quality_threshold:
			return False
			
		return True
	
	def _get_face_size(self, bbox):
		"""Calculate face size from bounding box"""
		top, right, bottom, left = bbox
		return (right - left) * (bottom - top)
	
	def _save_face_image(self, face, original_frame, faces_dir, face_id, frame_info):
		"""Save individual face image and return metadata with filename prepend"""
		try:
			bbox = face['bbox']
			embedding = face.get('embedding')
			top, right, bottom, left = bbox
			
			# Extract face from original high-resolution frame
			face_img = original_frame[top:bottom, left:right]
			if face_img.size == 0:
				return None
			
			# Create filename with prepend from original filename
			original_filename = Path(frame_info['filename']).stem  # Remove extension
			face_filename = f"{original_filename}_face_{face_id:05d}_{bbox}.jpg"
			face_path = faces_dir / face_filename
			
			# Use optimized JPEG encoding
			encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85,  # Good quality/speed balance
						   int(cv2.IMWRITE_JPEG_OPTIMIZE), 1]   # Optimize file size
			
			success = cv2.imwrite(str(face_path), face_img, encode_param)
			if not success:
				return None
			
			return {
				'face_id': face_id,
				'filename': face_filename,
				'original_filename': frame_info['filename'],
				'frame_number': frame_info['frame_number'],
				'embedding': embedding,
				'bbox': bbox,
				'confidence': face.get('confidence', 1.0),
				'source_file_path': frame_info.get('source_file_path'),  # Path to original frame file
				'video_name': frame_info.get('video_name')  # Video name if from video
			}
		except Exception as e:
			logger.error(f"Error saving face {face_id}: {e}")
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
				timestamp, frame_number, top, right, bottom, left, ext = match.groups()
				timestamp = float(timestamp)
				bbox = (int(top), int(right), int(bottom), int(left))
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
						top, right, bottom, left = bbox
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


def optimized_clustering_pipeline(faces_data, max_workers=None):
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
	for name, _, elapsed in successful_runs:
		logger.info(f"  {name}: {elapsed:.2f}s")


if __name__ == "__main__":
	# --- Configuration ---
	videoPath = "/home/jebineinstein/git/CaptionCreator/reuse/The Reader 2008_compressed/The Reader 2008_compressed_split_video_3.mp4"
	source_path = "/home/jebineinstein/git/TextFrameAligner/temp_dir/frames/"  # Can be video file or frames folder
	detect_type = "dinov3"
	
	# Performance settings
	frame_skip = 10  # Increased for faster processing
	batch_size = 16  # Smaller batches for memory efficiency
	max_faces_per_frame = 8  # Limit faces per frame
	min_face_size = 40  # Filter small faces
	quality_threshold = 0.6  # Filter low-quality detections
	use_gpu_decode = False  # Disable GPU decode to avoid format issues
	num_workers = None
	
	# --- Select Detector ---
	if detect_type == "insight_face":
		from insight_face_manager import InsightFaceManager
		detect_manager = InsightFaceManager()
	elif detect_type == "face_recognition":
		from face_recognition_manager import FaceRecognitionManager
		detect_manager = FaceRecognitionManager()
	elif detect_type == "deepface":
		from deepface_manager import DeepFaceManager
		detect_manager = DeepFaceManager()
	elif detect_type == "dinov3":
		from dinov3_manager import FaceDINOManager
		detect_manager = FaceDINOManager()
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
	processor = VideoProcessor(detect_manager, num_workers=num_workers)
	
	start_total = time.time()
	faces_data = []
	
	# Use the new unified method that handles both videos and frame folders
	faces_data = processor.extract_faces_from_source(
		# source_path,
		videoPath,
		frame_skip=frame_skip,
		batch_size=batch_size,
		max_faces_per_frame=max_faces_per_frame,
		min_face_size=min_face_size,
		quality_threshold=quality_threshold,
		use_gpu_decode=use_gpu_decode
	)

	optimized_clustering_pipeline(faces_data)
	
	total_elapsed = time.time() - start_total
	logger.info(f"\nTotal pipeline completed in {total_elapsed:.2f} seconds")
	logger.info(f"Processed {len(faces_data)} faces")
	processor.annotate_video(videoPath)