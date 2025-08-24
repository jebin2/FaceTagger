import cv2
import torch
import numpy as np
from PIL import Image
import face_recognition
from detect_manager import DetectManager


class MoonDream2FaceManager(DetectManager):
	"""
	DetectManager implementation using Moondream2 for face detection.
	Returns a list of dicts:
	  {
		'bbox': (top, right, bottom, left),		# pixel coords
		'score': float or None,					# detector confidence if available
		'embedding': np.ndarray or None			# 128-d face_recognition embedding if found
	  }
	"""

	def __init__(self):
		super().__init__()
		self.model = None

	def _load_model(self):
		from transformers import AutoModelForCausalLM

		self.model = AutoModelForCausalLM.from_pretrained(
			"vikhyatk/moondream2",
			revision="2025-06-21",
			trust_remote_code=True,
			device_map={"": "cuda"}  # ...or 'mps', on Apple Silicon
		)

	def _detect_frame(self, frame, file_path=None):
		"""
		Detect faces in a frame using Moondream2.
		Returns list of dicts: {'bbox': (top, right, bottom, left), 'score': float|None, 'embedding': array|None}
		"""
		if self.model is None:
			raise RuntimeError("Model not loaded. Call load() first.")

		pil_img = Image.open(file_path)
		w, h = pil_img.size
		face_data = []
		with torch.inference_mode():
			for obj in self.model.detect(pil_img, "face")["objects"]:
				left   = int(obj['x_min'] * w)
				top	   = int(obj['y_min'] * h)
				right  = int(obj['x_max'] * w)
				bottom = int(obj['y_max'] * h)

				# Crop for embedding (face_recognition expects RGB)
				crop = pil_img.crop((left, top, right, bottom))
				crop_rgb = np.array(crop)  # already RGB
				
				# Get crop dimensions
				crop_height, crop_width = crop_rgb.shape[:2]

				crop_face_location = (0, crop_width, crop_height, 0)
				
				# face_encodings expects a LIST of face locations
				encs = face_recognition.face_encodings(crop_rgb, known_face_locations=[crop_face_location])
				embedding = encs[0] if len(encs) > 0 else None
				face_data.append({
					'bbox': (left, top, right, bottom),
					'embedding': embedding
				})

		return face_data

	def _cleanup_model(self):
		"""Cleanup Moondream2 resources."""
		try:
			self.model = None
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		finally:
			print("MoonDream2FaceManager cleanup done.")

# if __name__ == "__main__":
# 	moondream2 = MoonDream2FaceManager()
# 	print(moondream2.detect(frame=None, file_path="all_frames_dir/frame_78.0_00780.jpg"))
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

    detect_manager = MoonDream2FaceManager(is_anime=is_anime)

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
