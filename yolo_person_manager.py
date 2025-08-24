import cv2
from detect_manager import DetectManager
from ultralytics import YOLO
import face_recognition

class YOLOPersonManager(DetectManager):
    """
    DetectManager implementation for YOLO person detection.
    Automatically loads and cleans up.
    Returns embeddings using face_recognition on detected faces.
    """
    def __init__(self, model_path='yolo12m.pt', conf_threshold=0.5):
        super().__init__()
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None

    def _load_model(self):
        """Load YOLO model."""
        print("Loading YOLO model...")
        self.model = YOLO(self.model_path)
        print("YOLO model loaded.")

    def _detect_frame(self, frame, file_path=None):
        """
        Detect people in a frame using YOLO.
        Returns list of dicts: {'bbox': (top, right, bottom, left), 'normed_embedding': array}
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        results = self.model.predict(frame, imgsz=640, conf=self.conf_threshold, classes=[0])  # 0 = person class
        face_data = []

        for r in results:
            for box in r.boxes:
                # YOLO returns [x1, y1, x2, y2, conf, cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                top, left, bottom, right = y1, x1, y2, x2

                person_img = frame[top:bottom, left:right]
                if person_img.size == 0:
                    continue

                # Optional: compute embedding from person region (e.g., face)
                # Convert to RGB
                rgb_crop = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                # Use face_recognition to extract embedding if a face is detected
                locations = face_recognition.face_locations(rgb_crop)
                embeddings = face_recognition.face_encodings(rgb_crop, locations) if locations else [None]

                # If multiple faces detected, take the first one
                embedding = embeddings[0] if embeddings else None

                face_data.append({
                    'bbox': (top, right, bottom, left),
                    'embedding': embedding
                })

        return face_data

    def _cleanup_model(self):
        """Cleanup YOLO resources."""
        self.model = None
        print("YOLOPersonManager cleanup done.")
