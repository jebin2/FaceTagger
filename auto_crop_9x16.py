"""
Smart Video Crop System
Automatically crops horizontal videos to 9:16 vertical format by intelligently 
tracking faces, persons, objects, and motion. Then applies the crop to generate
the final vertical video.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import json
from typing import List, Tuple, Optional
from dataclasses import dataclass
from custom_logger import logger_config


@dataclass
class CropConfig:
    """Configuration for video cropping parameters"""
    target_width: int = 1080
    target_height: int = 1920
    
    # Smoothing parameters
    smooth_alpha: float = 0.12
    deadzone: int = 20
    max_jump: int = 35
    disappear_tolerance: int = 15
    
    # Scene detection
    scene_change_threshold: int = 28
    
    # Objects to track (COCO class IDs)
    object_classes: List[int] = None
    
    def __post_init__(self):
        if self.object_classes is None:
            self.object_classes = [0, 1, 2, 3, 16, 17, 18]  # person, bike, car, motorcycle, dog, horse, sheep


class DetectionModels:
    """Manages YOLO detection models"""
    
    def __init__(self, face_model_path: str = "yolov12l-face.pt", 
                 person_model_path: str = "yolo12l.pt"):
        self.face_model = YOLO(face_model_path)
        self.person_model = YOLO(person_model_path)
    
    def detect_faces(self, frame: np.ndarray) -> np.ndarray:
        """Detect faces in frame, returns bounding boxes"""
        results = self.face_model(frame, verbose=False, device="cpu")
        if results[0].boxes is not None:
            return results[0].boxes.xyxy.cpu().numpy()
        return np.array([])
    
    def detect_persons_and_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect persons and objects, returns boxes and class IDs"""
        results = self.person_model(frame, verbose=False)
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            return boxes, classes
        return np.array([]), np.array([])


class SceneDetector:
    """Detects scene changes using histogram differences"""
    
    def __init__(self, threshold: int = 28):
        self.threshold = threshold
        self.prev_gray = None
    
    def is_scene_change(self, current_gray: np.ndarray) -> bool:
        """Check if current frame represents a scene change"""
        if self.prev_gray is None:
            self.prev_gray = current_gray
            return False
        
        diff = cv2.absdiff(self.prev_gray, current_gray)
        diff_val = diff.mean()
        self.prev_gray = current_gray
        
        return diff_val > self.threshold


class CropPositionCalculator:
    """Calculates optimal crop position based on detected subjects"""
    
    def __init__(self, config: CropConfig, models: DetectionModels):
        self.config = config
        self.models = models
        self.missing_frames = 0
    
    def calculate_from_faces(self, faces: np.ndarray) -> Optional[int]:
        """Calculate center X from face detections"""
        if len(faces) == 0:
            return None
        
        self.missing_frames = 0
        centers = [(box[0] + box[2]) / 2 for box in faces]
        widths = [box[2] - box[0] for box in faces]
        
        # Single face - center on it
        if len(faces) == 1:
            return int(centers[0])
        
        # Multiple faces - try to include all or prioritize largest
        areas = [(i, widths[i] * widths[i]) for i in range(len(faces))]
        areas.sort(key=lambda x: x[1], reverse=True)
        
        primary_idx = areas[0][0]
        primary_center = centers[primary_idx]
        
        min_center = min(centers)
        max_center = max(centers)
        span_width = max_center - min_center
        
        # All faces fit in frame
        if span_width <= self.config.target_width:
            return int((min_center + max_center) / 2)
        
        # Too wide - follow primary with 80/20 weight toward secondary
        secondary_idx = areas[1][0]
        secondary_center = centers[secondary_idx]
        return int(primary_center * 0.8 + secondary_center * 0.2)
    
    def calculate_from_persons(self, boxes: np.ndarray, classes: np.ndarray) -> Optional[int]:
        """Calculate center X from person detections"""
        person_centers = []
        for i, cls in enumerate(classes):
            if int(cls) == 0:  # COCO person class
                x1, y1, x2, y2 = boxes[i]
                person_centers.append((x1 + x2) / 2)
        
        if len(person_centers) > 0:
            return int(sum(person_centers) / len(person_centers))
        return None
    
    def calculate_from_objects(self, boxes: np.ndarray, classes: np.ndarray) -> Optional[int]:
        """Calculate center X from tracked object detections"""
        object_centers = []
        for i, cls in enumerate(classes):
            if int(cls) in self.config.object_classes:
                x1, y1, x2, y2 = boxes[i]
                object_centers.append((x1 + x2) / 2)
        
        if len(object_centers) > 0:
            return int(sum(object_centers) / len(object_centers))
        return None
    
    def calculate_from_motion(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> Optional[int]:
        """Calculate center X from optical flow motion"""
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        motion_mask = cv2.threshold(mag, 0.5, 1, cv2.THRESH_BINARY)[1]
        xs = np.where(motion_mask == 1)[1]
        
        if len(xs) > 0:
            return int(np.mean(xs))
        return None


class SmoothingFilter:
    """Applies smoothing to crop position transitions"""
    
    def __init__(self, config: CropConfig):
        self.config = config
    
    def smooth(self, current: int, previous: int) -> int:
        """Apply deadzone, max jump, and EMA smoothing"""
        diff = current - previous
        
        # Apply deadzone
        if abs(diff) < self.config.deadzone:
            current = previous
        
        # Apply max jump clamp
        if diff > self.config.max_jump:
            current = previous + self.config.max_jump
        elif diff < -self.config.max_jump:
            current = previous - self.config.max_jump
        
        # Apply exponential moving average
        current = int(self.config.smooth_alpha * current + 
                     (1 - self.config.smooth_alpha) * previous)
        
        return current


class SmartCropProcessor:
    """Main processor for smart video cropping analysis"""
    
    def __init__(self, video_path: str, output_json: str, config: CropConfig = None):
        self.video_path = video_path
        self.output_json = output_json
        self.config = config or CropConfig()
        
        self.models = DetectionModels()
        self.scene_detector = SceneDetector(self.config.scene_change_threshold)
        self.position_calc = CropPositionCalculator(self.config, self.models)
        self.smoother = SmoothingFilter(self.config)
        
        # Video properties
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.orig_w = 0
        self.orig_h = 0
        
        # Processing state
        self.crop_positions = []
        self.prev_center_x = 0
        self.prev_gray = None
    
    def initialize_video(self):
        """Open video and read properties"""
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.prev_center_x = self.orig_w // 2
        
        print(f"Video: {self.orig_w}x{self.orig_h} @ {self.fps}fps, {self.frame_count} frames")
    
    def process_frame(self, frame: np.ndarray, gray: np.ndarray) -> int:
        """Process single frame and return crop center X position"""
        
        # Check for scene change
        if self.scene_detector.is_scene_change(gray):
            self.prev_center_x = self.orig_w // 2
            self.position_calc.missing_frames = 0
            print(f"Scene change detected, resetting to center")
        
        # Priority 1: Face detection
        faces = self.models.detect_faces(frame)
        center_x = self.position_calc.calculate_from_faces(faces)
        
        if center_x is not None:
            return center_x
        
        # No faces detected
        self.position_calc.missing_frames += 1
        
        # Wait before switching logic
        if self.position_calc.missing_frames < self.config.disappear_tolerance:
            return self.prev_center_x
        
        # Priority 2: Person detection
        boxes, classes = self.models.detect_persons_and_objects(frame)
        center_x = self.position_calc.calculate_from_persons(boxes, classes)
        
        if center_x is not None:
            return center_x
        
        # Priority 3: Other tracked objects
        center_x = self.position_calc.calculate_from_objects(boxes, classes)
        
        if center_x is not None:
            return center_x
        
        # Priority 4: Motion-based fallback
        if self.prev_gray is not None:
            center_x = self.position_calc.calculate_from_motion(self.prev_gray, gray)
            if center_x is not None:
                return center_x
        
        # Final fallback: keep previous position
        return self.prev_center_x
    
    def calculate_crop_left(self, center_x: int) -> int:
        """Calculate left edge of crop from center X position"""
        left = max(0, center_x - self.config.target_width // 2)
        left = min(left, self.orig_w - self.config.target_width)
        return left
    
    def process_video(self):
        """Process entire video and generate crop positions"""
        self.initialize_video()
        
        print("Processing frames...")
        for frame_idx in range(self.frame_count):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate raw center position
            center_x = self.process_frame(frame, gray)
            
            # Apply smoothing
            center_x = self.smoother.smooth(center_x, self.prev_center_x)
            
            # Calculate final crop position
            left = self.calculate_crop_left(center_x)
            self.crop_positions.append(left)
            
            # Update state
            self.prev_center_x = center_x
            self.prev_gray = gray
            
            logger_config.info(f"Processed {frame_idx + 1}/{self.frame_count} frames", overwrite=True)
        
        self.cap.release()
        print("Processing complete!")
    
    def save_crop_data(self):
        """Save crop positions to JSON file"""
        data = {
            "fps": self.fps,
            "width": self.orig_w,
            "height": self.orig_h,
            "crop_left": self.crop_positions
        }
        
        with open(self.output_json, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Crop data saved: {self.output_json}")
    
    def run(self):
        """Execute complete processing pipeline"""
        self.process_video()
        self.save_crop_data()


class VideoRenderer:
    """Applies crop positions to video and renders final output"""
    
    def __init__(self, video_path: str, crop_json: str, output_path: str, 
                 target_width: int = 1080, target_height: int = 1920):
        self.video_path = video_path
        self.crop_json = crop_json
        self.output_path = output_path
        self.target_width = target_width
        self.target_height = target_height
        
        self.crop_positions = []
        self.fps = 0
        self.orig_w = 0
        self.orig_h = 0
    
    def load_crop_data(self):
        """Load crop positions from JSON file"""
        with open(self.crop_json, "r") as f:
            data = json.load(f)
        
        self.crop_positions = data["crop_left"]
        print(f"Loaded {len(self.crop_positions)} crop positions from {self.crop_json}")
    
    def render_video(self):
        """Apply crops and render final vertical video"""
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Rendering {frame_count} frames at {self.fps} fps...")
        print(f"Output size: {self.target_width}x{self.target_height}")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            self.fps, 
            (self.target_width, self.target_height)
        )
        
        frame_id = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get crop position for this frame
            left = self.crop_positions[frame_id]
            right = left + self.target_width
            
            # Ensure valid bounds
            left = max(0, min(left, self.orig_w - self.target_width))
            right = left + self.target_width
            
            # Crop horizontally
            cropped = frame[:, left:right]
            
            # Resize to target dimensions
            resized = cv2.resize(
                cropped, 
                (self.target_width, self.target_height), 
                interpolation=cv2.INTER_AREA
            )
            
            out.write(resized)
            frame_id += 1
            
            if (frame_id) % 100 == 0:
                print(f"Rendered {frame_id}/{frame_count} frames")
        
        cap.release()
        out.release()
        
        print(f"Final vertical video saved: {self.output_path}")
    
    def run(self):
        """Execute complete rendering pipeline"""
        self.load_crop_data()
        self.render_video()


class SmartCropPipeline:
    """Complete pipeline for analyzing and rendering smart-cropped videos"""
    
    def __init__(self, video_path: str, crop_json: str = "crop_data.json", 
                 output_video: str = "output_vertical.mp4", config: CropConfig = None):
        self.video_path = video_path
        self.crop_json = crop_json
        self.output_video = output_video
        self.config = config or CropConfig()
    
    def run_analysis(self):
        """Run crop position analysis"""
        print("=" * 60)
        print("STEP 1: Analyzing video for optimal crop positions")
        print("=" * 60)
        processor = SmartCropProcessor(self.video_path, self.crop_json, self.config)
        processor.run()
    
    def run_rendering(self):
        """Run video rendering with crop positions"""
        print("\n" + "=" * 60)
        print("STEP 2: Rendering final vertical video")
        print("=" * 60)
        renderer = VideoRenderer(
            self.video_path, 
            self.crop_json, 
            self.output_video,
            self.config.target_width,
            self.config.target_height
        )
        renderer.run()
    
    def run_full_pipeline(self):
        """Execute complete analysis and rendering pipeline"""
        self.run_analysis()
        self.run_rendering()
        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("=" * 60)


def main(VIDEO_PATH):
    """Main entry point"""
    CROP_JSON = "crop_data.json"
    OUTPUT_VIDEO = "output_vertical.mp4"
    
    config = CropConfig(
        target_width=1080,
        target_height=1920,
        smooth_alpha=0.12,
        deadzone=20,
        max_jump=35,
        disappear_tolerance=15,
        scene_change_threshold=28
    )
    
    # Run complete pipeline
    pipeline = SmartCropPipeline(VIDEO_PATH, CROP_JSON, OUTPUT_VIDEO, config)
    pipeline.run_full_pipeline()
    
    # Or run steps individually:
    # pipeline.run_analysis()
    # pipeline.run_rendering()


if __name__ == "__main__":
    VIDEO_PATH = "/home/jebin/git/CaptionCreator/tempOutput/gywuRYSyDn_captioned_movie_shorts.mp4"
    main(VIDEO_PATH)