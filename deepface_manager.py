import os
import cv2
import shutil
from deepface import DeepFace
from detect_manager import DetectManager


class DeepFaceManager(DetectManager):
    """
    DetectManager implementation using DeepFace (ArcFace).
    Returns list of dicts: {'bbox': (x, y, w, h), 'embedding': array}
    """

    def __init__(self, model_name="ArcFace", detector_backend="retinaface", similarity_threshold=0.5, min_face_ratio=0.01):
        super().__init__()
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.similarity_threshold = similarity_threshold
        self.min_face_ratio = min_face_ratio
        self.face_images = []
        self.groups_as_sets = []

    def _load_model(self):
        print(f"DeepFaceManager ready. Using {self.model_name} with {self.detector_backend}")

    def _detect_frame(self, img_path):
        """
        Detect all faces in an image and return list of dicts with bbox + embedding.
        Returns [] if no face detected.
        """
        faces = []
        try:
            reps = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True
            )
            if not reps:
                return []

            # image size
            img = cv2.imread(img_path)
            if img is None:
                return []

            img_area = img.shape[0] * img.shape[1]

            for rep in reps:
                bbox = rep.get("facial_area")
                if not bbox:
                    continue
                x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
                face_area = w * h
                if (face_area / img_area) < self.min_face_ratio:
                    continue

                faces.append({
                    "bbox": (x, y, w, h),
                    "embedding": rep["embedding"]
                })

            return faces
        except Exception:
            return []

    def _cleanup_model(self):
        print("DeepFaceManager cleanup done.")
