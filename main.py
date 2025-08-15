import time
import traceback
import cv2
import json
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

# Configure logging for better progress tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
	def __init__(self, detect_manager: DetectManager, num_workers=None):
		self.detect_manager = detect_manager
		self.num_workers = num_workers or min(cpu_count(), 8)  # Cap at 8 to avoid memory issues
		
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
		faces_dir = Path("extracted_faces")
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
		
		# --- Load model once ---
		self.detect_manager.load()
		
		# --- Process video with optimizations ---
		all_face_data = self._process_video_parallel(
			cap, fps, total_frames, frame_skip, batch_size, 
			faces_dir, max_faces_per_frame, min_face_size, quality_threshold
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
		with open(metadata_file, 'w') as f:
			json.dump(data_clean, f, indent=2)

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
	
	def _process_video_parallel(self, cap, fps, total_frames, frame_skip, batch_size, faces_dir, max_faces_per_frame, min_face_size, quality_threshold):
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
		
		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				break
				
			# Handle different frame formats that might come from video decoder
			if frame is None:
				frame_count += 1
				continue
				
			# Ensure frame is in correct format (BGR)
			if len(frame.shape) == 2:  # Grayscale
				frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
			elif len(frame.shape) == 3 and frame.shape[2] == 1:  # Single channel
				frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
			elif len(frame.shape) == 3 and frame.shape[2] == 4:  # BGRA
				frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
				
			if frame_count % frame_skip == 0:
				# Smart frame skipping based on motion
				if self._should_process_frame(frame, prev_frame_gray, motion_threshold):
					# Resize frame for faster processing if it's too large
					processed_frame = self._preprocess_frame(frame)
					frame_buffer.append(processed_frame)
					frame_info_buffer.append({'original_frame': frame, 'frame_number': frame_count})
					
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
								face_counter, frame_info['frame_number']
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
	
	def _save_face_image(self, face, original_frame, faces_dir, face_id, frame_number):
		"""Save individual face image and return metadata"""
		try:
			bbox = face['bbox']
			embedding = face.get('embedding')
			top, right, bottom, left = bbox
			
			# Extract face from original high-resolution frame
			face_img = original_frame[top:bottom, left:right]
			if face_img.size == 0:
				return None
			
			# Save with optimized JPEG settings for speed
			face_filename = f"face_{face_id:05d}.jpg"
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
				'frame_number': frame_number,
				'embedding': embedding,
				'bbox': bbox,
				'confidence': face.get('confidence', 1.0)
			}
		except Exception as e:
			logger.error(f"Error saving face {face_id}: {e}")
			return None
	
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
	"""Optimized version with parallel file operations"""
	output_dir = Path("output") / method_name
	
	if output_dir.exists():
		shutil.rmtree(output_dir)
	output_dir.mkdir(parents=True)

	def copy_face_file(args):
		face, label = args
		group_name = f"person_{label:03d}" if label != -1 else "unknown"
		group_path = output_dir / group_name
		group_path.mkdir(exist_ok=True)
		
		source_path = Path("extracted_faces") / face['filename']
		if source_path.exists():
			shutil.copy2(source_path, group_path)  # copy2 preserves metadata

	# Parallel file copying
	with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
		executor.map(copy_face_file, zip(face_data, labels))
	
	# Print summary
	label_counts = Counter(labels)
	num_persons = len(label_counts) - (1 if -1 in label_counts else 0)
	num_unknown = label_counts.get(-1, 0)
	
	logger.info(f"Results for {method_name}: {num_persons} persons, {num_unknown} unknown faces")
	logger.info(f"Saved to: {output_dir}")


def optimized_clustering_pipeline(faces_data, max_workers=None):
	"""Run clustering algorithms in parallel where possible"""
	from group_face import (cluster_with_hdbscan, cluster_with_spectral, 
						   cluster_with_adaptive_similarity, cluster_with_agglomerative, 
						   cluster_with_affinity_propagation, cluster_with_chinese_whispers)
	
	face_encodings = [face['embedding'] for face in faces_data if face['embedding'] is not None]
	
	if not face_encodings:
		logger.warning("No face embeddings found for clustering")
		return
	
	# Convert to numpy array for better performance
	face_encodings = np.array(face_encodings)
	
	clustering_techniques = [
		("hdbscan", lambda: cluster_with_hdbscan(face_encodings)),
		("spectral_clustering", lambda: cluster_with_spectral(face_encodings)),
		("adaptive_similarity", lambda: cluster_with_adaptive_similarity(face_encodings)),
		("agglomerative", lambda: cluster_with_agglomerative(face_encodings)),
		("affinity_propagation", lambda: cluster_with_affinity_propagation(face_encodings)),
		("chinese_whispers", lambda: cluster_with_chinese_whispers(face_encodings)),
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
	video_path = "input.mp4"
	detect_type = "insight_face"
	
	# Performance settings
	frame_skip = 100  # Increased for faster processing
	batch_size = 16  # Smaller batches for memory efficiency
	max_faces_per_frame = 8  # Limit faces per frame
	min_face_size = 40  # Filter small faces
	quality_threshold = 0.6  # Filter low-quality detections
	use_gpu_decode = False  # Disable GPU decode to avoid format issues
	
	# --- Select Detector ---
	if detect_type == "insight_face":
		from insight_face_manager import InsightFaceManager
		detect_manager = InsightFaceManager()
	elif detect_type == "face_recognition":
		from face_recognition_manager import FaceRecognitionManager
		detect_manager = FaceRecognitionManager(model='cnn')
	elif detect_type == "mediapipe":
		from mediapipe_face_manager import MediaPipeFaceManager
		detect_manager = MediaPipeFaceManager(model_selection=1, min_detection_confidence=0.5)
	elif detect_type == "yolo":
		from yolo_person_manager import YOLOPersonManager
		detect_manager = YOLOPersonManager(conf_threshold=0.5)
	else:
		raise ValueError(f"Unknown detect_type: {detect_type}")

	# --- Run Optimized Pipeline ---
	processor = VideoProcessor(detect_manager)
	
	start_total = time.time()
	faces_data = processor.extract_faces_from_video_optimized(
		video_path, 
		frame_skip=frame_skip,
		batch_size=batch_size,
		max_faces_per_frame=max_faces_per_frame,
		min_face_size=min_face_size,
		quality_threshold=quality_threshold,
		use_gpu_decode=use_gpu_decode
	)
	
	if faces_data:
		optimized_clustering_pipeline(faces_data)
	
	total_elapsed = time.time() - start_total
	logger.info(f"\nTotal pipeline completed in {total_elapsed:.2f} seconds")
	logger.info(f"Processed {len(faces_data)} faces")