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
                #  dinov3_model="facebook/dinov3-vits16-pretrain-lvd1689m", 
                 dinov3_model="facebook/dinov3-vitl16-pretrain-lvd1689m", 
                 use_patch_features=False):
        super().__init__()
        self.face_model = face_model
        self.dinov3_model = dinov3_model
        self.use_patch_features = use_patch_features
        self.processor = None
        self.model = None
        self._embedding_cache = {}  # 🔹 cache for embeddings
        self._normalized_embedding_cache = {}  # 🔹 cache for embeddings

    def _get_embedding_from_cache(self, image_path):
        """Return cached embedding if exists, else compute & store"""
        if image_path in self._embedding_cache:
            return self._embedding_cache[image_path]

        img = Image.open(image_path).convert("RGB")
        embedding = self._extract_dino_embedding(img)
        self._embedding_cache[image_path] = embedding
        return embedding

    def _load_model(self):
        """Load DINOv3 model & processor"""
        if self.model is None or self.processor is None:
            print(f"Loading DINOv3 model: {self.dinov3_model}")
            self.processor = AutoImageProcessor.from_pretrained(self.dinov3_model)
            self.model = AutoModel.from_pretrained(self.dinov3_model)
            self.model.eval()
            print("FaceDINOManager ready.")

    def _load_yolo_model(self):
        from ultralytics import YOLO
        """Load YOLO model for person detection."""
        if not hasattr(self, "yolo_model") or self.yolo_model is None:
            self.yolo_model = YOLO("yolov12l-face.pt")  # change path if needed

    def get_person_bboxes(self, frame):
        """
        Detect persons using YOLO and return bounding boxes
        in (top, right, bottom, left) format.
        """
        # Ensure YOLO model is loaded
        self._load_yolo_model()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.yolo_model.predict(rgb_frame, verbose=False)
        bboxes = []

        for result in results:
            for box in result.boxes:
                # cls = int(box.cls[0])
                # if cls != 0:  # class 0 = person in COCO
                #     continue

                # Convert xyxy -> (top, right, bottom, left)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                top, right, bottom, left = y1, x2, y2, x1
                bboxes.append((top, right, bottom, left))

        return bboxes

    def _extract_dino_embedding(self, face_image):
        """Extract embedding from a cropped face using DINOv3"""
        if self.model is None or self.processor is None:
            self._load_model()

        inputs = self.processor(images=face_image, return_tensors="pt")
        with torch.inference_mode():
            outputs = self.model(**inputs)

        # Use pooler_output instead of raw features
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embedding = outputs.pooler_output.squeeze().numpy()
        else:
            # Fallback to your current approach
            features = outputs.last_hidden_state
            if self.use_patch_features:
                patch_features = features[:, 1:, :]
                embedding = torch.mean(patch_features, dim=1).squeeze().numpy()
            else:
                cls_features = features[:, 0, :]
                embedding = cls_features.squeeze().numpy()

        return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

    def add_padding_to_bbox(self, bbox, img_shape, padding_ratio=0.15):
        top, right, bottom, left = bbox
        h, w = img_shape[:2]

        # Calculate padding
        pad_h = int((bottom - top) * padding_ratio)
        pad_w = int((right - left) * padding_ratio)

        # Expand and clip within image
        top = max(0, top - pad_h)
        bottom = min(h, bottom + pad_h)
        left = max(0, left - pad_w)
        right = min(w, right + pad_w)

        return top, right, bottom, left

    def _detect_frame(self, frame, file_path=None):
        """
        Detect faces and extract embeddings with DINOv3.
        """
        if self.model is None or self.processor is None:
            self._load_model()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # face_locations = face_recognition.face_locations(rgb_frame, model=self.face_model)
        face_locations = self.get_person_bboxes(frame)

        faces = []
        for (top, right, bottom, left) in face_locations:
            top, right, bottom, left = self.add_padding_to_bbox((top, right, bottom, left), frame.shape, padding_ratio = 0)
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

    def compute_similarity(self, image_path_1, image_path_2):
        """Compute cosine similarity between two images using DINO embeddings"""
        if self.model is None or self.processor is None:
            self._load_model()

        # 🔹 Use cached embeddings
        features1_norm = self.get_normalized_embedding_from_cache(image_path_1)
        features2_norm = self.get_normalized_embedding_from_cache(image_path_2)

        # Cosine similarity
        # similarity = torch.mm(features1_norm, features2_norm.T)
        # Simple dot product since vectors are already normalized
        # This is much faster than torch.mm for single comparisons
        similarity = torch.dot(features1_norm.squeeze(), features2_norm.squeeze())

        return similarity.item()

    def get_normalized_embedding_from_cache(self, image_path):
        if image_path in self._normalized_embedding_cache:
            return self._normalized_embedding_cache[image_path]

        features = self._get_embedding_from_cache(image_path)

        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Normalize
        features_norm = torch.nn.functional.normalize(features_tensor, p=2, dim=-1)
        self._normalized_embedding_cache[image_path] = features_norm
        return features_norm
