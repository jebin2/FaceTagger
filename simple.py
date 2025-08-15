import time
import os
import cv2
import json
from pathlib import Path
import shutil
from collections import Counter
from detect_manager import DetectManager


def extract_faces_from_video(video_path, detect_manager: DetectManager, frame_skip=10):
	"""
	Extracts all faces from a video using a given DetectManager, saves them as images,
	and stores their embeddings. Shows progress with elapsed/estimated time, video timestamp,
	current frame, and number of faces found.
	"""
	if not isinstance(detect_manager, DetectManager):
		raise ValueError("detect_manager must be an instance of DetectManager")

	# --- Directory and Metadata Setup ---
	faces_dir = "extracted_faces"
	shutil.rmtree(faces_dir)
	print(f"--- Starting Face Extraction with '{detect_manager.__class__.__name__}' from '{video_path}' ---")
	
	Path(faces_dir).mkdir(parents=True, exist_ok=True)
	metadata_file = Path(faces_dir) / 'face_metadata.json'
	if metadata_file.exists():
		print(f"Faces already extracted. Loading from '{metadata_file}'.")
		with open(metadata_file, 'r') as f:
			return json.load(f)

	# --- Video Capture Setup ---
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(f"Error: Could not open video file {video_path}")
		return []

	fps = cap.get(cv2.CAP_PROP_FPS)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	total_seconds = total_frames / fps if fps > 0 else 0

	all_face_data = []
	face_counter = 0
	frame_count = 0

	# --- Load the model ---
	detect_manager.load()
	start_time = time.time()

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		if frame_count % frame_skip == 0:
			detected_faces = detect_manager.detect(frame)

			for face in detected_faces:
				bbox = face['bbox']
				embedding = face.get('embedding')
				top, right, bottom, left = bbox

				face_img = frame[top:bottom, left:right]
				if face_img.size == 0:
					continue

				face_filename = f"face_{face_counter:05d}.jpg"
				cv2.imwrite(os.path.join(faces_dir, face_filename), face_img)

				all_face_data.append({
					'face_id': face_counter,
					'filename': face_filename,
					'frame_number': frame_count,
					'embedding': embedding.tolist() if embedding is not None else None
				})
				face_counter += 1

			# --- Progress Update ---
			elapsed = time.time() - start_time
			estimated_total = (elapsed / (frame_count + 1)) * total_frames
			elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
			est_min, est_sec = divmod(int(estimated_total / fps), 60)

			current_sec = frame_count / fps
			current_min, current_s = divmod(int(current_sec), 60)
			total_min, total_s = divmod(int(total_seconds), 60)

			print(
				f"Frame: {frame_count}/{total_frames} | "
				f"Video Time: {current_min:02d}:{current_s:02d}/{total_min:02d}:{total_s:02d} | "
				f"Faces: {len(all_face_data)} | "
				f"Elapsed: {elapsed_min:02d}:{elapsed_sec:02d} / Est. Total: {est_min:02d}:{est_sec:02d}",
				end='\r'
			)

		frame_count += 1

	print("\n" + "-"*30)
	cap.release()
	detect_manager.cleanup()

	with open(metadata_file, 'w') as f:
		json.dump(all_face_data, f, indent=2)

	print(f"--- Face Extraction Complete ---")
	print(f"Extracted {len(all_face_data)} faces and saved them to '{faces_dir}'.\n")
	
	return all_face_data

def save_grouped_faces(face_data, labels, method_name):
	"""Saves the clustered face images into labeled folders."""
	output_dir = Path("output") / method_name
	
	# Remove old results if they exist
	if output_dir.exists():
		shutil.rmtree(output_dir)
	output_dir.mkdir(parents=True)

	# Group faces by their assigned label
	for i, face in enumerate(face_data):
		label = labels[i]
		
		# Determine the group folder name
		# Noise points from DBSCAN (label -1) are put in an "unknown" folder
		group_name = f"person_{label:03d}" if label != -1 else "unknown"
		group_path = output_dir / group_name
		group_path.mkdir(exist_ok=True)
		
		# Copy the original face image to the new group folder
		source_path = os.path.join("extracted_faces", face['filename'])
		if os.path.exists(source_path):
			shutil.copy(source_path, group_path)
	
	# Print a summary of the clustering results
	label_counts = Counter(labels)
	num_persons = len(label_counts) - (1 if -1 in label_counts else 0)
	num_unknown = label_counts.get(-1, 0)
	
	print(f"--- Results for {method_name} ---")
	print(f"Found {num_persons} unique persons and {num_unknown} unknown faces.")
	print(f"Results saved to: '{output_dir}'")
	print("-" * 30 + "\n")

def initial_grouping_technique(faces_data):
	from group_face import cluster_with_hdbscan, cluster_with_spectral, cluster_with_adaptive_similarity, cluster_with_agglomerative, cluster_with_affinity_propagation, cluster_with_chinese_whispers
	 # Get all encodings from the extracted data
	face_encodings = [face['encoding'] for face in faces_data]

	# 2. LIST OF TECHNIQUES
	# Each entry is a tuple: (name_for_output_folder, function_to_call)
	clustering_techniques = [
		("hdbscan", lambda: cluster_with_hdbscan(face_encodings)),
		# ("mean_shift", lambda: cluster_with_mean_shift(face_encodings)), # not good at all
		("spectral_clustering", lambda: cluster_with_spectral(face_encodings)),
		# ("approximate_rank_order", lambda: cluster_with_approximate_rank_order(face_encodings)), # not good at all
		("adaptive_similarity", lambda: cluster_with_adaptive_similarity(face_encodings)),
		# ("dbscan", lambda: cluster_with_dbscan(face_encodings)), # not good at all
		("agglomerative", lambda: cluster_with_agglomerative(face_encodings)),
		("affinity_propagation", lambda: cluster_with_affinity_propagation(face_encodings)),
		("chinese_whispers", lambda: cluster_with_chinese_whispers(face_encodings)),
		# ("similarity_threshold", lambda: cluster_with_similarity(face_encodings)) # not good at all
	]

	# 3. LOOP THROUGH TECHNIQUES
	print("\n--- Starting Face Grouping with Different Techniques ---")
	for name, technique_func in clustering_techniques:
		# Run the clustering algorithm
		cluster_labels = technique_func()
		
		# Save the results in a dedicated folder
		save_grouped_faces(faces_data, cluster_labels, name)

	print("=== All Processing Complete! ===")

if __name__ == "__main__":
	# --- Configuration ---
	video_path = "input.mp4"
	detect_type = "insight_face"  # Options: insight_face, face_recognition, mediapipe, yolo

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

	# --- Extract Faces ---
	faces_data = extract_faces_from_video(video_path, detect_manager=detect_manager, frame_skip=10)

	initial_grouping_technique(faces_data)

	print(f"\nTotal faces extracted: {len(faces_data)}")