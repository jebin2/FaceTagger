from abc import ABC, abstractmethod

class DetectManager(ABC):
	"""
	Abstract base class for face detectors.
	Automatically handles lazy loading and cleanup.
	"""
	def __init__(self):
		self._loaded = False

	def load(self):
		"""Lazy load model if not loaded yet."""
		if not self._loaded:
			self._load_model()
			self._loaded = True

	@abstractmethod
	def _load_model(self):
		"""Subclass should implement actual model loading."""
		pass

	def detect(self, frame):
		"""Detect faces, automatically loading model if needed."""
		if not self._loaded:
			self.load()
		return self._detect_frame(frame)

	@abstractmethod
	def _detect_frame(self, frame):
		"""Subclass should implement actual frame detection."""
		pass

	def cleanup(self):
		"""Cleanup model and memory."""
		if self._loaded:
			self._cleanup_model()
			self._loaded = False

	@abstractmethod
	def _cleanup_model(self):
		"""Subclass should implement model cleanup."""
		pass

	def __del__(self):
		"""Automatic cleanup on deletion."""
		self.cleanup()
