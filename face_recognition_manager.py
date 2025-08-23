import cv2
import face_recognition
from detect_manager import DetectManager
import time
import numpy as np

class FaceRecognitionManager(DetectManager):
	"""
	DetectManager implementation using `face_recognition` library.
	Returns list of dicts: {'bbox': (top, right, bottom, left), 'embedding': array}
	"""
	def __init__(self, model='cnn'):
		super().__init__()
		self.model = model  # 'hog' or 'cnn'
		self.loaded = False
		self._load_model()

	def _load_model(self):
		"""
		Pre-warm dlib by running a dummy detection.
		"""
		print("Loading FaceRecognition model (%s)...", self.model)
		start = time.time()
		# Run a dummy detection with a 1x1 image to warm up dlib & CUDA
		dummy = cv2.cvtColor((255 * np.ones((10, 10, 3), dtype=np.uint8)), cv2.COLOR_BGR2RGB)
		face_recognition.face_locations(dummy, model=self.model)
		self.loaded = True
		print("Model loaded in %.2fs", time.time() - start)

	def _detect_frame(self, frame):
		"""
		Detect faces in a frame and return list of dicts:
		[{'bbox': (top, right, bottom, left), 'embedding': array}, ...]
		"""
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# Detect face locations and compute embeddings
		face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
		face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

		faces = []
		for loc, enc in zip(face_locations, face_encodings):
			# loc is already (top, right, bottom, left)
			faces.append({'bbox': loc, 'embedding': enc.tolist()})
		return faces

	def _cleanup_model(self):
		"""
		Nothing to cleanup for face_recognition, included for interface consistency.
		"""
		print("FaceRecognitionManager cleanup done.")

if __name__ == "__main__":
	import sys
	import json
	from tqdm import tqdm
	if len(sys.argv) < 2:
		print("Usage: python face_recognition_manager.py <image_path>")
		sys.exit(1)

	images_path = sys.argv[1]
	with  open(images_path, 'r') as f:
		images_path_obj = json.load(f)
	detect_manager = FaceRecognitionManager()

	for obj in tqdm(images_path_obj, desc="Detecting Faces"):
		frame = cv2.imread(obj["frame_path"])
		if "face_location" not in obj:
			obj.update({
				"face_location": None
			})
		if "already_processed_face" not in obj:
			obj.update({
				"already_processed_face": False
			})

		if not obj["already_processed_face"]:
			result = detect_manager._detect_frame(frame)
			obj.update({
				"already_processed_face": True
			})
			if result:
				obj.update({
					"face_location": result[0]['bbox']
				})

			with open(images_path, 'w') as f:
				json.dump(images_path_obj, f, indent=4)