import cv2
import face_recognition
from detect_manager import DetectManager
import time
import numpy as np
from ultralytics import YOLO


class FaceRecognitionManager(DetectManager):
    """
    DetectManager implementation.
    Supports real faces (face_recognition) and anime faces (YOLO).
    Always returns list of dicts:
    [ {'bbox': (left, top, right, bottom), 'embedding': array or None}, ... ]
    """

    def __init__(self, model_name='cnn', is_anime=False, anime_model_path="yolov8x6_animeface.pt"):
        super().__init__()
        self.is_anime = is_anime
        self.model_name = model_name  # 'hog' or 'cnn'
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
            # Warm up dlib (face_recognition)
            dummy = cv2.cvtColor((255 * np.ones((10, 10, 3), dtype=np.uint8)), cv2.COLOR_BGR2RGB)
            face_recognition.face_locations(dummy, model=self.model_name)

        self.loaded = True
        print("Model loaded in %.2fs" % (time.time() - start))

    def _detect_frame(self, frame, file_path=None):
        """
        Detect faces in a frame and return list of dicts:
        [ {'bbox': (left, top, right, bottom), 'embedding': array or None}, ... ]
        """
        faces = []

        if self.is_anime:
            # YOLO returns (x1, y1, x2, y2) -> (left, top, right, bottom)
            results = self.model(frame, verbose=False)
            for result in results:
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    bbox = (x1, y1, x2, y2)
                    faces.append({'bbox': bbox, 'embedding': None})
        else:
            # face_recognition returns (top, right, bottom, left)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model=self.model_name)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
                bbox = (left, top, right, bottom)  # convert to (l, t, r, b)
                faces.append({'bbox': bbox, 'embedding': enc.tolist()})

        return faces

    def _cleanup_model(self):
        print("FaceRecognitionManager cleanup done.")


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

    detect_manager = FaceRecognitionManager(is_anime=is_anime)

    for obj in tqdm(images_path_obj, desc="Detecting Faces"):
        frame = cv2.imread(obj["frame_path"])
        if "face_location" not in obj:
            obj.update({"face_location": None})
        if "already_processed_face" not in obj:
            obj.update({"already_processed_face": False})

        if not obj["already_processed_face"]:
            result = detect_manager._detect_frame(frame)
            obj.update({"already_processed_face": True})
            if result:
                obj.update({"face_location": result[0]['bbox']})

            with open(images_path, 'w') as f:
                json.dump(images_path_obj, f, indent=4)
