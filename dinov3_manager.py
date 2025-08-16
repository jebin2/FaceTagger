import os
import cv2
import torch
import numpy as np
from PIL import Image
import face_recognition
from transformers import AutoImageProcessor, AutoModel
from detect_manager import DetectManager


class FaceDINOManager(DetectManager):
    """
    DetectManager that uses face_recognition for detection 
    and DINOv3 for face embeddings.
    Returns list of dicts: {'bbox': (top, right, bottom, left), 'embedding': array}
    """

    def __init__(self, 
                 face_model="cnn", 
                 dinov3_model="facebook/dinov3-vits16-pretrain-lvd1689m", 
                 use_patch_features=False):
        super().__init__()
        self.face_model = face_model
        self.dinov3_model = dinov3_model
        self.use_patch_features = use_patch_features
        self.processor = None
        self.model = None

    def _load_model(self):
        """Load DINOv3 model & processor"""
        if self.model is None or self.processor is None:
            print(f"Loading DINOv3 model: {self.dinov3_model}")
            self.processor = AutoImageProcessor.from_pretrained(self.dinov3_model)
            self.model = AutoModel.from_pretrained(self.dinov3_model)
            self.model.eval()
            print("FaceDINOManager ready.")

    def _extract_dino_embedding(self, face_image):
        """Extract embedding from a cropped face using DINOv3"""
        if self.model is None or self.processor is None:
            self._load_model()

        inputs = self.processor(images=face_image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        features = outputs.last_hidden_state
        if self.use_patch_features:
            # Average over patch features
            patch_features = features[:, 1:, :]  # exclude CLS
            embedding = torch.mean(patch_features, dim=1).squeeze().numpy()
        else:
            # CLS token
            cls_features = features[:, 0, :]
            embedding = cls_features.squeeze().numpy()

        return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

    def _detect_frame(self, frame):
        """
        Detect faces and extract embeddings with DINOv3.
        """
        if self.model is None or self.processor is None:
            self._load_model()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model=self.face_model)

        faces = []
        for (top, right, bottom, left) in face_locations:
            # Crop face
            face_crop = rgb_frame[top:bottom, left:right]
            if face_crop.size == 0:
                continue
            face_pil = Image.fromarray(face_crop)

            # Extract embedding with DINOv3
            embedding = self._extract_dino_embedding(face_pil)

            faces.append({
                "bbox": (top, right, bottom, left),
                "embedding": embedding
            })

        return faces

    def _cleanup_model(self):
        """Cleanup resources"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("FaceDINOManager cleanup done.")

    def get_similarity(self, encodings, threshold: float = 0.6):
        """
        Cluster embeddings based on cosine similarity.
        """
        if len(encodings) == 0:
            return []

        encodings_np = np.array(encodings)
        n = len(encodings_np)
        labels = [-1] * n
        current_label = 0

        for i in range(n):
            if labels[i] != -1:
                continue
            labels[i] = current_label
            for j in range(i + 1, n):
                if labels[j] == -1:
                    sim = float(np.dot(encodings_np[i], encodings_np[j]))
                    if sim >= threshold:
                        labels[j] = current_label
            current_label += 1

        return labels
