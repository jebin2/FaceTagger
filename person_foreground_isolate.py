"""
Person Segmentation Background Processor
Segments persons from video using YOLO and applies background effects
(blur or custom image) while keeping the foreground sharp.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import os
from typing import Optional, List, Tuple
from dataclasses import dataclass
from custom_logger import logger_config

@dataclass
class BackgroundConfig:
    """Configuration for background processing"""
    enable_background_blur: bool = True
    bg_image_path: Optional[str] = None
    enable_pipe_output: bool = True
    blur_kernel_size: Tuple[int, int] = (55, 55)
    mask_threshold: float = 0.3


@dataclass
class SmoothingConfig:
    """Configuration for tracking smoothing"""
    smooth_alpha: float = 0.15
    deadzone: int = 10
    max_jump: int = 40


class PersonSegmentationModel:
    """Manages YOLO segmentation model for person detection"""
    
    def __init__(self, model_path: str = "yolov8s-seg.pt"):
        print(f"Loading YOLO-Seg: {model_path}")
        self.model = YOLO(model_path)
    
    def segment_persons(self, frame: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Detect and segment persons in frame
        Returns: (union_mask, list_of_centers)
        """
        results = self.model(frame, device="cpu", verbose=False)[0]
        
        masks = results.masks
        boxes = results.boxes
        
        person_masks = []
        centers = []
        
        if boxes is not None:
            classes = boxes.cls.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            for i, cls in enumerate(classes):
                if int(cls) == 0:  # COCO person class
                    m = masks.data[i].cpu().numpy()
                    person_masks.append(m)
                    
                    x1, y1, x2, y2 = xyxy[i]
                    centers.append((x1 + x2) / 2)
        
        return person_masks, centers
    
    def create_union_mask(self, person_masks: List[np.ndarray], 
                         frame_width: int, frame_height: int,
                         threshold: float = 0.3) -> np.ndarray:
        """Create combined mask from multiple person masks"""
        if len(person_masks) == 0:
            return np.zeros((frame_height, frame_width), dtype=np.uint8)
        
        final_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        
        for mask in person_masks:
            resized_mask = cv2.resize(mask, (frame_width, frame_height))
            binary_mask = (resized_mask > threshold).astype(np.uint8) * 255
            final_mask = np.maximum(final_mask, binary_mask)
        
        return final_mask


class TrackingSmoothing:
    """Applies smoothing to person tracking positions"""
    
    def __init__(self, config: SmoothingConfig):
        self.config = config
        self.prev_center_x = None
    
    def smooth_position(self, current_x: float, default_x: float) -> int:
        """Apply smoothing with deadzone, max jump, and EMA"""
        if self.prev_center_x is None:
            self.prev_center_x = current_x
        
        diff = current_x - self.prev_center_x
        
        # Apply deadzone - ignore tiny movements
        if abs(diff) < self.config.deadzone:
            current_x = self.prev_center_x
        
        # Apply max jump clamp
        if diff > self.config.max_jump:
            current_x = self.prev_center_x + self.config.max_jump
        elif diff < -self.config.max_jump:
            current_x = self.prev_center_x - self.config.max_jump
        
        # Apply exponential moving average
        current_x = (self.config.smooth_alpha * current_x + 
                    (1 - self.config.smooth_alpha) * self.prev_center_x)
        
        self.prev_center_x = int(current_x)
        return int(current_x)
    
    def reset(self):
        """Reset tracking state"""
        self.prev_center_x = None


class BackgroundProcessor:
    """Processes video backgrounds with blur or custom images"""
    
    def __init__(self, config: BackgroundConfig):
        self.config = config
        self.custom_bg = None
        
        if config.bg_image_path and os.path.isfile(config.bg_image_path):
            self.custom_bg = cv2.imread(config.bg_image_path)
            print(f"Loaded custom background: {config.bg_image_path}")
    
    def create_background(self, frame: np.ndarray, 
                         width: int, height: int) -> np.ndarray:
        """Create background layer (blurred or custom image)"""
        if self.custom_bg is not None:
            bg = cv2.resize(self.custom_bg, (width, height))
        else:
            bg = frame.copy()
        
        if self.config.enable_background_blur:
            bg = cv2.GaussianBlur(bg, self.config.blur_kernel_size, 0)
        
        return bg
    
    def composite_frame(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Composite foreground and background using mask"""
        # Extract foreground using mask
        fg = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Create background
        height, width = frame.shape[:2]
        bg = self.create_background(frame, width, height)
        
        # Extract background using inverted mask
        inv_mask = cv2.bitwise_not(mask)
        bg_part = cv2.bitwise_and(bg, bg, mask=inv_mask)
        
        # Combine foreground and background
        final_frame = cv2.add(bg_part, fg)
        
        return final_frame


class VideoOutputWriter:
    """Handles video output via FFmpeg pipe or frame files"""
    
    def __init__(self, output_path: str, width: int, height: int, 
                 fps: int, use_pipe: bool = True):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.use_pipe = use_pipe
        self.ffmpeg_process = None
        self.frame_dir = "frames"
        
        if self.use_pipe:
            self._init_ffmpeg_pipe()
        else:
            self._init_frame_output()
    
    def _init_ffmpeg_pipe(self):
        """Initialize FFmpeg pipe for fast video output"""
        self.ffmpeg_process = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", str(self.fps),
                "-i", "-",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                self.output_path
            ],
            stdin=subprocess.PIPE
        )
        print(f"FFmpeg pipe initialized for {self.output_path}")
    
    def _init_frame_output(self):
        """Initialize frame-by-frame file output"""
        os.makedirs(self.frame_dir, exist_ok=True)
        print(f"Frame output directory: {self.frame_dir}")
    
    def write_frame(self, frame: np.ndarray, frame_id: int):
        """Write single frame to output"""
        if self.use_pipe:
            self.ffmpeg_process.stdin.write(frame.tobytes())
        else:
            cv2.imwrite(f"{self.frame_dir}/{frame_id:05d}.png", frame)
    
    def close(self):
        """Close output writer"""
        if self.use_pipe and self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            print(f"Video saved: {self.output_path}")


class PersonSegmentationProcessor:
    """Main processor for person segmentation and background effects"""
    
    def __init__(self, input_video: str, output_video: str,
                 model_path: str = "yolov8s-seg.pt",
                 bg_config: BackgroundConfig = None,
                 smooth_config: SmoothingConfig = None):
        self.input_video = input_video
        self.output_video = output_video
        
        self.bg_config = bg_config or BackgroundConfig()
        self.smooth_config = smooth_config or SmoothingConfig()
        
        # Initialize components
        self.model = PersonSegmentationModel(model_path)
        self.smoother = TrackingSmoothing(self.smooth_config)
        self.bg_processor = BackgroundProcessor(self.bg_config)
        
        # Video properties
        self.cap = None
        self.writer = None
        self.fps = 0
        self.width = 0
        self.height = 0
        self.frame_count = 0
    
    def initialize_video(self):
        """Open input video and read properties"""
        self.cap = cv2.VideoCapture(self.input_video)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {self.width}x{self.height}, {self.fps} FPS, {self.frame_count} frames")
        
        # Initialize output writer
        self.writer = VideoOutputWriter(
            self.output_video,
            self.width,
            self.height,
            self.fps,
            self.bg_config.enable_pipe_output
        )
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Process single frame with segmentation and background effects
        Returns: (processed_frame, center_x)
        """
        # Segment persons
        person_masks, centers = self.model.segment_persons(frame)
        
        # Create union mask
        mask = self.model.create_union_mask(
            person_masks,
            self.width,
            self.height,
            self.bg_config.mask_threshold
        )
        
        # Calculate center position
        if len(centers) > 0:
            center_x = float(np.mean(centers))
        else:
            # No person detected - use previous or default
            center_x = self.smoother.prev_center_x if self.smoother.prev_center_x else self.width // 2
        
        # Apply smoothing
        center_x = self.smoother.smooth_position(center_x, self.width // 2)
        
        # Composite frame with background effects
        final_frame = self.bg_processor.composite_frame(frame, mask)
        
        return final_frame, center_x
    
    def process_video(self):
        """Process entire video with segmentation and effects"""
        self.initialize_video()
        
        print("Processing frames...")
        frame_id = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            final_frame, center_x = self.process_frame(frame)
            
            # Write output
            self.writer.write_frame(final_frame, frame_id)
            
            frame_id += 1
            
            if frame_id % 100 == 0:
                logger_config.info(f"Processed {frame_id}/{self.frame_count} frames")
        
        # Cleanup
        self.cap.release()
        self.writer.close()
        
        print("Processing complete!")
    
    def run(self):
        """Execute complete processing pipeline"""
        self.process_video()


def main():
    """Main entry point"""
    INPUT_VIDEO = "output_from_auto_crop_9x16.mp4"
    OUTPUT_VIDEO = "final_simple_edit.mp4"
    MODEL_PATH = "yolo11l-seg.pt"
    
    # Configure background effects
    bg_config = BackgroundConfig(
        enable_background_blur=True,
        bg_image_path=None,  # Set to image path for custom background
        enable_pipe_output=True,
        blur_kernel_size=(55, 55),
        mask_threshold=0.3
    )
    
    # Configure smoothing
    smooth_config = SmoothingConfig(
        smooth_alpha=0.15,
        deadzone=10,
        max_jump=40
    )
    
    # Run processor
    processor = PersonSegmentationProcessor(
        INPUT_VIDEO,
        OUTPUT_VIDEO,
        MODEL_PATH,
        bg_config,
        smooth_config
    )
    
    processor.run()


if __name__ == "__main__":
    main()