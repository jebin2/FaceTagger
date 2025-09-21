import cv2
from insightface.app import FaceAnalysis
from detect_manager import DetectManager
from ultralytics import YOLO
import time


class InsightFaceManager(DetectManager):
	"""
	DetectManager implementation for InsightFace or YOLO (anime).
	Returns list of {'bbox': (left, top, right, bottom), 'embedding': array or None}.
	"""

	def __init__(self, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], is_anime=False, anime_model_path="yolov8x6_animeface.pt"):
		super().__init__()
		self.providers = providers
		self.app = None
		self.is_anime = is_anime
		self.anime_model_path = anime_model_path
		self.loaded = False
		self.model = None
		self._load_model()

	def _load_model(self):
		print("Loading FaceRecognition model...")
		start = time.time()

		if self.is_anime:
			# Load YOLO anime model
			self.model = YOLO(self.anime_model_path)
		else:
			print("Loading InsightFace models...")
			self.app = FaceAnalysis(name='buffalo_l', providers=self.providers)
			self.app.prepare(ctx_id=0, det_size=(640, 640))
			print("InsightFace models loaded.")

		self.loaded = True
		print("Model loaded in %.2fs" % (time.time() - start))

	def _detect_frame(self, frame, file_path=None):
		"""
		Detect faces and return list of dicts with bounding boxes and embeddings.
		"""
		faces_data = []

		if self.is_anime:
			results = self.model(frame, verbose=False)
			for result in results:
				for box in result.boxes.xyxy.cpu().numpy():
					x1, y1, x2, y2 = map(int, box[:4])
					faces_data.append({
						'bbox': (x1, y1, x2, y2),  # (left, top, right, bottom)
						'embedding': None
					})
		else:
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			faces = self.app.get(frame_rgb)
			for face in faces:
				xmin, ymin, xmax, ymax = map(int, face.bbox)
				faces_data.append({
					'bbox': (xmin, ymin, xmax, ymax),
					'embedding': face.normed_embedding.tolist(),
				})

		return faces_data

	def _cleanup_model(self):
		print("Cleaning up InsightFace models...")
		self.app = None
		print("Cleanup done.")

if __name__ == "__main__":
	import sys
	import json
	from tqdm import tqdm

	if len(sys.argv) < 3:
		print("Usage: python face_recognition_manager.py <image_path.json> <anime|real>")
		sys.exit(1)

	images_path = sys.argv[1]
	mode = sys.argv[2].lower()
	is_anime = (mode == "anime")

	with open(images_path, 'r') as f:
		images_path_obj = json.load(f)

	detect_manager = InsightFaceManager(is_anime=is_anime)

	for obj in tqdm(images_path_obj, desc="Detecting Faces"):
		frame = cv2.imread(obj["frame_path"])
		
		if "face_location" not in obj:
			obj["face_location"] = None
		if "already_processed_face" not in obj:
			obj["already_processed_face"] = False

		if not obj["already_processed_face"]:
			results = detect_manager._detect_frame(frame)  # assuming this returns list of dicts with 'bbox'
			obj["already_processed_face"] = True

			if results:
				obj["all_face"] = [face['bbox'] for face in results]
				# Find the largest face by area
				largest_face = max(
					results, 
					key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1])
				)
				obj["face_location"] = largest_face['bbox']  # left, top, right, bottom

			with open(images_path, 'w') as f:
				json.dump(images_path_obj, f, indent=4)
