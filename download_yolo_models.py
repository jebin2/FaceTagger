#!/usr/bin/env python3
import os
import requests

MODELS_DIR = "models"
MIN_SIZE_MB = 5

MODELS = {
    "yolo11l-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt",
    "yolo12l.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12l.pt",
    "yolo12m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12m.pt",
    "yolov12l-face.pt": "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12l-face.pt",
    "yolov8s-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-seg.pt",
    "yolov8x6_animeface.pt": "https://huggingface.co/Fuyucchi/yolov8_animeface/resolve/main/yolov8x6_animeface.pt",
}

os.makedirs(MODELS_DIR, exist_ok=True)


def valid(path):
    return os.path.exists(path) and os.path.getsize(path) > MIN_SIZE_MB * 1024 * 1024


def download(name, url):
    path = os.path.join(MODELS_DIR, name)

    if valid(path):
        print(f"✅ {name} already exists")
        return

    print(f"⬇️ Downloading {name}")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    if not valid(path):
        os.remove(path)
        raise RuntimeError(f"❌ {name} download failed or file too small")

    size = os.path.getsize(path) / (1024 * 1024)
    print(f"✅ Saved {name} ({size:.1f} MB)")


def download_all_models():
    print(f"Checking {len(MODELS)} models...")
    for name, url in MODELS.items():
        download(name, url)
    print("\n🎉 All YOLO models check/download complete")

if __name__ == "__main__":
    download_all_models()
