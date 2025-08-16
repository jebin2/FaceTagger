import cv2
from insightface.app import FaceAnalysis
from detect_manager import DetectManager

class InsightFaceManager(DetectManager):
	"""
	DetectManager implementation for InsightFace.
	Returns list of {'bbox': (top, right, bottom, left), 'embedding': array}.
	"""
	def __init__(self, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
		super().__init__()
		self.providers = providers
		self.app = None

	def _load_model(self):
		print("Loading InsightFace models...")
		self.app = FaceAnalysis(name='buffalo_l', providers=self.providers)
		self.app.prepare(ctx_id=0, det_size=(640, 640))
		print("InsightFace models loaded.")

	def _detect_frame(self, frame):
		"""
		Detect faces and return list of dicts with bounding boxes and embeddings.
		"""
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		faces = self.app.get(frame_rgb)

		face_data = []
		for face in faces:
			# Convert InsightFace bbox (xmin, ymin, xmax, ymax) to (top, right, bottom, left)
			bbox = face.bbox.astype(int)
			top, left, bottom, right = bbox[1], bbox[0], bbox[3], bbox[2]
			face_data.append({
				'bbox': (top, right, bottom, left),
				'embedding': face.normed_embedding.tolist(),  # high-quality 512-d embedding
			})
		return face_data

	def _cleanup_model(self):
		print("Cleaning up InsightFace models...")
		self.app = None
		print("Cleanup done.")
