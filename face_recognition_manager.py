import cv2
import face_recognition
from detect_manager import DetectManager

class FaceRecognitionManager(DetectManager):
	"""
	DetectManager implementation using `face_recognition` library.
	Returns list of dicts: {'bbox': (top, right, bottom, left), 'embedding': array}
	"""
	def __init__(self, model='cnn'):
		super().__init__()
		self.model = model  # 'hog' or 'cnn'
		self.loaded = True  # face_recognition does not require heavy loading

	def _load_model(self):
		"""
		No heavy model to load for face_recognition; placeholder for interface consistency.
		"""
		print("FaceRecognitionManager ready.")

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
