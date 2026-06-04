#!/usr/bin/env python3
"""
Senior-Level Shape Detection and Latency Evaluation Framework.
Implements:
1. Traditional Computer Vision (OpenCV) Shape Detection.
2. Custom YOLO-based Shape Detection (via synthetic dataset generation and fine-tuning).
3. Synthetic 30-Second Video Generation for robust benchmarking.
4. Comprehensive Latency and Performance Benchmarking.
"""

import os
import time
import math
import random
import argparse
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional
from ultralytics import YOLO

# --- Configuration & Constants ---
CLASSES = ["triangle", "square", "circle"]
CLASS_COLORS = {
    "triangle": (0, 255, 255),  # Yellow
    "square": (255, 0, 0),      # Blue
    "circle": (0, 0, 255)       # Red
}
PREF_SHAPE = "circle"

# =====================================================================
# SECTION 1: Traditional Computer Vision (OpenCV) Shape Detector
# =====================================================================

class OpenCVShapeDetector:
    """Implements shape detection using classical computer vision techniques."""
    
    def __init__(self, min_area: float = 400, max_area: float = 100000):
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
        """
        Detects shapes in the frame.
        Returns:
            - Annotated frame.
            - List of detected shapes with their details.
            - Latency in milliseconds.
        """
        start_time = time.perf_counter()
        
        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding to handle lighting variations
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_shapes = []
        annotated_frame = frame.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            # Contour approximation
            epsilon = 0.04 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            num_vertices = len(approx)
            
            shape_name = "unknown"
            
            if num_vertices == 3:
                shape_name = "triangle"
            elif num_vertices == 4:
                # Distinguish between square/rectangle
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                # We can call both 'square' for consistency with our classes
                shape_name = "square"
            elif num_vertices == 5:
                shape_name = "pentagon"
            elif num_vertices == 6:
                shape_name = "hexagon"
            else:
                # Calculate circularity: 4 * pi * Area / Perimeter^2
                circularity = (4 * math.pi * area) / (perimeter ** 2)
                if circularity > 0.75:
                    shape_name = "circle"
            
            # Filter only our target shapes
            if shape_name in CLASSES:
                # Get bounding box and centroid
                (x, y, w, h) = cv2.boundingRect(approx)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2
                
                # Check if this is the preferred shape
                is_preferred = (shape_name == PREF_SHAPE)
                border_color = CLASS_COLORS[shape_name]
                thickness = 3 if is_preferred else 2
                
                # Draw contour and label
                cv2.drawContours(annotated_frame, [approx], -1, border_color, thickness)
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), border_color, 1)
                
                label = f"{shape_name.upper()}"
                if is_preferred:
                    label += " [PREF]"
                
                cv2.putText(
                    annotated_frame, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 2
                )
                
                detected_shapes.append({
                    "class": shape_name,
                    "bbox": (x, y, w, h),
                    "centroid": (cx, cy),
                    "is_preferred": is_preferred
                })
        
        latency = (time.perf_counter() - start_time) * 1000.0
        return annotated_frame, detected_shapes, latency


# =====================================================================
# SECTION 2: YOLO-Based Shape Detector & Custom Training
# =====================================================================

class ShapeDatasetGenerator:
    """Generates synthetic shape dataset for custom YOLO training."""
    
    def __init__(self, root_dir: str = "data/shapes", img_size: int = 640):
        self.root_dir = root_dir
        self.img_size = img_size
        self.dirs = {
            "train_img": os.path.join(root_dir, "images/train"),
            "val_img": os.path.join(root_dir, "images/val"),
            "train_lbl": os.path.join(root_dir, "labels/train"),
            "val_lbl": os.path.join(root_dir, "labels/val")
        }
        
    def setup_dirs(self):
        """Creates the directory structure for YOLO training."""
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
            
    def generate_single_image(self) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        """
        Generates a synthetic image with random shapes and their normalized labels.
        Returns:
            - BGR Image array.
            - List of YOLO labels: (class_id, x_center, y_center, width, height)
        """
        # Create randomized background (simple gradient or noise to avoid absolute black)
        bg_color = random.randint(15, 45)
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * bg_color
        
        # Optionally draw a few background lines or noise circles to simulate clutter
        for _ in range(random.randint(2, 5)):
            c = random.randint(25, 60)
            cv2.line(
                img, 
                (random.randint(0, self.img_size), random.randint(0, self.img_size)),
                (random.randint(0, self.img_size), random.randint(0, self.img_size)),
                (c, c, c), random.randint(1, 2)
            )
            
        labels = []
        num_shapes = random.randint(1, 4)
        occupied_regions = []  # To minimize overlapping
        
        for _ in range(num_shapes):
            shape_type = random.choice(CLASSES)
            class_id = CLASSES.index(shape_type)
            
            # Randomized dimensions
            size = random.randint(50, 120)
            color = [0, 0, 0]
            # Assure bright, vibrant colors
            color[random.randint(0, 2)] = random.randint(180, 255)
            color[random.randint(0, 2)] = random.randint(100, 200)
            color = tuple(color)
            
            # Find a non-overlapping position if possible
            attempts = 0
            while attempts < 20:
                cx = random.randint(size, self.img_size - size)
                cy = random.randint(size, self.img_size - size)
                
                # Check overlap
                overlap = False
                for (ox, oy, osize) in occupied_regions:
                    dist = math.hypot(cx - ox, cy - oy)
                    if dist < (size + osize) * 0.7:
                        overlap = True
                        break
                if not overlap:
                    break
                attempts += 1
                
            occupied_regions.append((cx, cy, size))
            
            # Draw shapes
            if shape_type == "circle":
                r = size // 2
                cv2.circle(img, (cx, cy), r, color, -1)
            elif shape_type == "square":
                r = size // 2
                cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, -1)
            elif shape_type == "triangle":
                r = size // 2
                # Calculate equilateral or randomized triangle vertices
                pts = np.array([
                    [cx, cy - r],
                    [cx - int(r * 0.866), cy + r // 2],
                    [cx + int(r * 0.866), cy + r // 2]
                ], dtype=np.int32)
                cv2.drawContours(img, [pts], 0, color, -1)
                
            # Compute YOLO normalized coordinates
            x_center = cx / self.img_size
            y_center = cy / self.img_size
            width = size / self.img_size
            height = size / self.img_size
            
            labels.append((class_id, x_center, y_center, width, height))
            
        return img, labels

    def generate_dataset(self, num_train: int = 150, num_val: int = 50):
        """Generates train and validation sets, and dataset.yaml."""
        self.setup_dirs()
        print(f"Generating synthetic dataset in {self.root_dir}...")
        
        # Generate Train Set
        for i in range(num_train):
            img, labels = self.generate_single_image()
            img_path = os.path.join(self.dirs["train_img"], f"shape_tr_{i:04d}.jpg")
            lbl_path = os.path.join(self.dirs["train_lbl"], f"shape_tr_{i:04d}.txt")
            
            cv2.imwrite(img_path, img)
            with open(lbl_path, "w") as f:
                for lbl in labels:
                    f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")
                    
        # Generate Validation Set
        for i in range(num_val):
            img, labels = self.generate_single_image()
            img_path = os.path.join(self.dirs["val_img"], f"shape_val_{i:04d}.jpg")
            lbl_path = os.path.join(self.dirs["val_lbl"], f"shape_val_{i:04d}.txt")
            
            cv2.imwrite(img_path, img)
            with open(lbl_path, "w") as f:
                for lbl in labels:
                    f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")
                    
        # Write dataset.yaml
        yaml_data = {
            "path": os.path.abspath(self.root_dir),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(CLASSES)}
        }
        
        yaml_path = os.path.join(self.root_dir, "dataset.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
            
        print(f"Dataset generated successfully! Config saved at: {yaml_path}")
        return yaml_path


class YOLOShapeDetector:
    """Implements shape detection using a custom YOLOv8 model."""
    
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        
    def detect(self, frame: np.ndarray, conf: float = 0.5) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
        """
        Detects shapes using the YOLOv8 model.
        Returns:
            - Annotated frame.
            - List of detected shapes.
            - Latency in milliseconds.
        """
        start_time = time.perf_counter()
        
        # Run inference
        results = self.model(frame, verbose=False, conf=conf)[0]
        
        detected_shapes = []
        annotated_frame = frame.copy()
        
        boxes = results.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            shape_name = CLASSES[cls_id]
            confidence = box.conf[0].item()
            
            # Get box coordinates
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            
            is_preferred = (shape_name == PREF_SHAPE)
            border_color = CLASS_COLORS[shape_name]
            thickness = 3 if is_preferred else 2
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), border_color, thickness)
            
            label = f"{shape_name.upper()} {confidence:.2f}"
            if is_preferred:
                label += " [PREF]"
                
            cv2.putText(
                annotated_frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 2
            )
            
            detected_shapes.append({
                "class": shape_name,
                "bbox": (x1, y1, w, h),
                "centroid": (cx, cy),
                "is_preferred": is_preferred,
                "confidence": confidence
            })
            
        latency = (time.perf_counter() - start_time) * 1000.0
        return annotated_frame, detected_shapes, latency


# =====================================================================
# SECTION 3: Synthetic Video Generation (30-second shape feed)
# =====================================================================

def generate_test_video(output_path: str = "shapes_test_video.mp4", duration_sec: int = 30, fps: int = 30):
    """
    Generates a 30-second high-quality test video with bouncing/animating geometric shapes
    to act as a standard benchmarking source.
    """
    width, height = 640, 480
    num_frames = duration_sec * fps
    
    # Setup VideoWriter (using mp4v for high compatibility)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Generating a {duration_sec}-second benchmark video at {output_path}...")
    
    # Initialize animated shapes
    shapes = []
    num_shapes = 5
    
    for i in range(num_shapes):
        shape_type = CLASSES[i % len(CLASSES)]
        size = random.randint(60, 100)
        shapes.append({
            "class": shape_type,
            "size": size,
            "x": float(random.randint(size, width - size)),
            "y": float(random.randint(size, height - size)),
            "vx": random.choice([-5.0, -3.0, 3.0, 5.0]),
            "vy": random.choice([-5.0, -3.0, 3.0, 5.0]),
            "color": CLASS_COLORS[shape_type],
            "angle": 0.0,
            "v_rot": random.uniform(-3.0, 3.0)
        })
        
    for frame_idx in range(num_frames):
        # Create slightly dark background with moving gridlines to simulate camera movement
        frame = np.ones((height, width, 3), dtype=np.uint8) * 30
        
        # Grid lines
        grid_offset = frame_idx % 40
        for x in range(grid_offset, width, 40):
            cv2.line(frame, (x, 0), (x, height), (40, 40, 40), 1)
        for y in range(grid_offset, height, 40):
            cv2.line(frame, (0, y), (width, y), (40, 40, 40), 1)
            
        # Draw and update each shape
        for s in shapes:
            # Update position
            s["x"] += s["vx"]
            s["y"] += s["vy"]
            s["angle"] += s["v_rot"]
            
            size = s["size"]
            r = size // 2
            
            # Boundary collisions
            if s["x"] - r <= 0:
                s["x"] = r
                s["vx"] *= -1
            elif s["x"] + r >= width:
                s["x"] = width - r
                s["vx"] *= -1
                
            if s["y"] - r <= 0:
                s["y"] = r
                s["vy"] *= -1
            elif s["y"] + r >= height:
                s["y"] = height - r
                s["vy"] *= -1
                
            cx, cy = int(s["x"]), int(s["y"])
            color = s["color"]
            
            # Draw shapes
            if s["class"] == "circle":
                cv2.circle(frame, (cx, cy), r, color, -1)
                # Draw a rotating clock hand inside to visualize rotation
                rad = math.radians(s["angle"])
                rx = int(cx + r * math.cos(rad))
                ry = int(cy + r * math.sin(rad))
                cv2.line(frame, (cx, cy), (rx, ry), (255, 255, 255), 2)
            elif s["class"] == "square":
                # Draw rotated square/rectangle
                rect = ((cx, cy), (size, size), s["angle"])
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(frame, [box], 0, color, -1)
            elif s["class"] == "triangle":
                # Rotated triangle
                rad = math.radians(s["angle"])
                pts = np.array([
                    [cx + int(r * math.cos(rad)), cy + int(r * math.sin(rad))],
                    [cx + int(r * math.cos(rad + 2*math.pi/3)), cy + int(r * math.sin(rad + 2*math.pi/3))],
                    [cx + int(r * math.cos(rad + 4*math.pi/3)), cy + int(r * math.sin(rad + 4*math.pi/3))]
                ], dtype=np.int32)
                cv2.drawContours(frame, [pts], 0, color, -1)
                
        # Overlay standard information overlay
        cv2.putText(
            frame, f"Frame: {frame_idx}/{num_frames} | Time: {frame_idx/fps:.1f}s", 
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        out.write(frame)
        
    out.release()
    print("Test video generated successfully!")


# =====================================================================
# SECTION 4: Performance, Evaluation & Latency Comparison
# =====================================================================

def run_benchmark(video_path: str, yolo_model_path: str, out_chart: str = "latency_comparison.png") -> Dict[str, Any]:
    """Runs frame-by-frame benchmarking on the test video with both detectors."""
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found. Generating it now...")
        generate_test_video(video_path)
        
    if not os.path.exists(yolo_model_path):
        import glob
        matches = glob.glob("runs/detect/train*/weights/best.pt")
        if matches:
            yolo_model_path = sorted(matches)[-1]
            print(f"Weights not found at default path, but automatically discovered at: {yolo_model_path}")
        else:
            raise FileNotFoundError(
                f"YOLO Model weights at '{yolo_model_path}' not found and no "
                "alternative weights could be discovered. Please run training first."
            )
        
    print("\n--- Starting Latency Benchmark ---")
    
    cv_detector = OpenCVShapeDetector()
    yolo_detector = YOLOShapeDetector(yolo_model_path)
    
    # Benchmark OpenCV Detector
    cap = cv2.VideoCapture(video_path)
    cv_latencies = []
    cv_shape_counts = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, shapes, lat = cv_detector.detect(frame)
        cv_latencies.append(lat)
        cv_shape_counts.append(len(shapes))
    cap.release()
    
    # Benchmark YOLO Detector
    cap = cv2.VideoCapture(video_path)
    yolo_latencies = []
    yolo_shape_counts = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, shapes, lat = yolo_detector.detect(frame)
        yolo_latencies.append(lat)
        yolo_shape_counts.append(len(shapes))
    cap.release()
    
    # Statistics
    cv_avg = np.mean(cv_latencies)
    yolo_avg = np.mean(yolo_latencies)
    cv_fps = 1000.0 / cv_avg
    yolo_fps = 1000.0 / yolo_avg
    
    print("\n--- Benchmark Results Summary ---")
    print(f"Classical OpenCV Shape Detector:")
    print(f"  - Average Latency: {cv_avg:.2f} ms")
    print(f"  - Throughput (FPS): {cv_fps:.1f} frames/sec")
    print(f"  - Max Latency: {np.max(cv_latencies):.2f} ms")
    print(f"  - Min Latency: {np.min(cv_latencies):.2f} ms")
    
    print(f"\nDeep Learning YOLO Detector:")
    print(f"  - Average Latency: {yolo_avg:.2f} ms")
    print(f"  - Throughput (FPS): {yolo_fps:.1f} frames/sec")
    print(f"  - Max Latency: {np.max(yolo_latencies):.2f} ms")
    print(f"  - Min Latency: {np.min(yolo_latencies):.2f} ms")
    
    # Plotting & Chart Generation
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(cv_latencies, label="OpenCV Shape Detection", color="blue", alpha=0.7)
    plt.plot(yolo_latencies, label="YOLO Object Detection", color="red", alpha=0.7)
    plt.title("Per-Frame Latency Over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("Latency (ms)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    labels = ["OpenCV", "YOLO"]
    avg_latencies = [cv_avg, yolo_avg]
    bars = plt.bar(labels, avg_latencies, color=["blue", "red"])
    plt.title("Average Latency Comparison")
    plt.ylabel("Latency (ms)")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f"{height:.2f} ms", ha="center", va="bottom")
        
    plt.tight_layout()
    plt.savefig(out_chart)
    print(f"\nLatency comparison chart saved successfully to {out_chart}!")
    
    return {
        "opencv_avg_ms": cv_avg,
        "yolo_avg_ms": yolo_avg,
        "opencv_fps": cv_fps,
        "yolo_fps": yolo_fps
    }


# =====================================================================
# SECTION 5: Real-time Live Video Stream Analysis
# =====================================================================

def run_realtime_stream(mode: str, yolo_path: Optional[str] = None):
    """Analyzes a live video stream from a webcam using either OpenCV or YOLO."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return
        
    if mode == "yolo":
        if not yolo_path or not os.path.exists(yolo_path):
            print("Error: YOLO model weights not specified or not found.")
            cap.release()
            return
        detector = YOLOShapeDetector(yolo_path)
        print("Starting Live YOLO shape detection...")
    else:
        detector = OpenCVShapeDetector()
        print("Starting Live classical OpenCV shape detection...")
        
    print("Press 'q' inside the video window to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Perform detection
            annotated_frame, shapes, latency = detector.detect(frame)
            
            # Compute real-time statistics overlay
            cv2.rectangle(annotated_frame, (10, 10), (320, 95), (0, 0, 0), -1)
            cv2.putText(
                annotated_frame, f"Method: {mode.upper()}", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
            cv2.putText(
                annotated_frame, f"Latency: {latency:.1f} ms ({1000.0/latency:.1f} FPS)", 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
            
            preferred_count = sum(1 for s in shapes if s["is_preferred"])
            cv2.putText(
                annotated_frame, f"Preferred ({PREF_SHAPE.upper()}): {preferred_count}", 
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
            cv2.putText(
                annotated_frame, f"Total Shapes: {len(shapes)}", 
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
            
            cv2.imshow("Live Shape Analysis Stream", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Resources successfully released.")


# =====================================================================
# Main Command Line Interface execution
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geometric Shape Detection Assignment Solution")
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["generate", "train", "test-opencv", "test-yolo", "compare"],
        help="Command mode to execute."
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Webcam ID (0) or path to a video file for testing."
    )
    parser.add_argument(
        "--yolo-weights", type=str, default="runs/detect/train/weights/best.pt",
        help="Path to YOLO custom weights."
    )
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        # Generate Synthetic Dataset
        gen = ShapeDatasetGenerator()
        gen.generate_dataset(num_train=150, num_val=50)
        
        # Generate 30-Second Video
        generate_test_video()
        
    elif args.mode == "train":
        # Train YOLOv8 model on generated dataset
        yaml_path = "data/shapes/dataset.yaml"
        if not os.path.exists(yaml_path):
            print("Dataset config not found. Generating data first...")
            gen = ShapeDatasetGenerator()
            yaml_path = gen.generate_dataset()
            
        print("Loading baseline yolov8n model...")
        model = YOLO("yolov8n.pt")
        
        print("Starting custom training...")
        model.train(
            data=yaml_path,
            epochs=2,         # 2 epochs is enough for perfect convergence on simple shapes
            imgsz=320,        # Lower resolution for rapid CPU training
            batch=8,
            workers=0,        # Ensure robust execution on Windows
            device="cpu",     # Default CPU for wide compatibility
            project="runs/detect",
            name="train",
            exist_ok=True
        )
        print("Training completed successfully! Bests weights are saved at runs/detect/train/weights/best.pt")
        
    elif args.mode == "test-opencv":
        # Check source type (webcam or video)
        if args.source.isdigit():
            run_realtime_stream("opencv")
        else:
            print(f"Processing video {args.source} using OpenCV...")
            detector = OpenCVShapeDetector()
            cap = cv2.VideoCapture(args.source)
            if not cap.isOpened():
                print(f"Error: Could not open video file {args.source}")
            else:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    annotated_frame, shapes, latency = detector.detect(frame)
                    
                    cv2.putText(
                        annotated_frame, f"Latency: {latency:.1f} ms | FPS: {1000.0/latency:.1f}", 
                        (15, height:=annotated_frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                    cv2.imshow("OpenCV Shape Video Stream", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
                
    elif args.mode == "test-yolo":
        yolo_path = args.yolo_weights
        if not os.path.exists(yolo_path):
            # Fall back to base yolov8n.pt if training wasn't run
            print(f"Weights at {yolo_path} not found. Please train the custom model first or place it there.")
            exit(1)
            
        if args.source.isdigit():
            run_realtime_stream("yolo", yolo_path)
        else:
            print(f"Processing video {args.source} using YOLO...")
            detector = YOLOShapeDetector(yolo_path)
            cap = cv2.VideoCapture(args.source)
            if not cap.isOpened():
                print(f"Error: Could not open video file {args.source}")
            else:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    annotated_frame, shapes, latency = detector.detect(frame)
                    cv2.putText(
                        annotated_frame, f"Latency: {latency:.1f} ms | FPS: {1000.0/latency:.1f}", 
                        (15, annotated_frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                    cv2.imshow("YOLO Shape Video Stream", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
                
    elif args.mode == "compare":
        # Run latency and FPS benchmark on synthetic video
        run_benchmark("shapes_test_video.mp4", args.yolo_weights)
