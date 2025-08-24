import cv2
import mediapipe as mp
import face_recognition
from detect_manager import DetectManager

class MediaPipeFaceManager(DetectManager):
	"""
	DetectManager implementation for MediaPipe Face Detection.
	Automatically loads and cleans up.
	Returns embeddings using face_recognition on detected faces.
	Only works reliably on real human faces, not anime.
	"""
	def __init__(self, model_selection=1, min_detection_confidence=0.5):
		super().__init__()
		self.model_selection = model_selection
		self.min_detection_confidence = min_detection_confidence
		self.face_detection = None
		self.mp_face_detection = mp.solutions.face_detection

	def _load_model(self):
		"""Initialize MediaPipe Face Detection."""
		self.face_detection = self.mp_face_detection.FaceDetection(
			model_selection=self.model_selection,
			min_detection_confidence=self.min_detection_confidence
		)
		print("MediaPipeFaceManager model loaded.")

	def _detect_frame(self, frame, file_path=None):
		"""
		Detect faces in a frame.
		Returns list of dicts: {'bbox': (top, right, bottom, left), 'normed_embedding': array}
		Only detections with confidence >= 0.8 are considered.
		"""
		if self.face_detection is None:
			raise RuntimeError("Model not loaded")

		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = self.face_detection.process(rgb_frame)

		face_data = []
		if results.detections:
			height, width, _ = frame.shape
			face_locations = []

			# Filter detections by confidence >= 0.8
			for detection in results.detections:
				score = detection.score[0] if detection.score else 0
				if score < 0.8:
					continue

				bbox = detection.location_data.relative_bounding_box
				top = int(bbox.ymin * height)
				left = int(bbox.xmin * width)
				bottom = int((bbox.ymin + bbox.height) * height)
				right = int((bbox.xmin + bbox.width) * width)
				face_locations.append((top, right, bottom, left))

			if face_locations:
				# Compute embeddings using face_recognition
				embeddings = face_recognition.face_encodings(rgb_frame, face_locations)
				for loc, emb in zip(face_locations, embeddings):
					face_data.append({'bbox': loc, 'embedding': emb})

		return face_data


	def _cleanup_model(self):
		"""Cleanup MediaPipe resources."""
		if self.face_detection:
			self.face_detection.close()
			self.face_detection = None
		print("MediaPipeFaceManager cleanup done.")
