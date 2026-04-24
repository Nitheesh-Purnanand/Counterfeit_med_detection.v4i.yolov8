"""
YOLOv8 Object Detection & Cropping for Counterfeit Medicine Detection
=====================================================================

This script demonstrates the FIRST stage of our pipeline:
  1. Load a fine-tuned YOLOv8 model (trained on authentic/counterfeit medicines)
  2. Run detection on all dataset images (train / valid / test splits)
  3. Crop each detected region and save it for downstream ML classification

The cropped images are used in `counterfeit_detection_augmented.ipynb` for
feature extraction and classifier training.

Usage
-----
    python yolo_detection_cropping.py          # Run detection on all splits
    python yolo_detection_cropping.py --split train   # Run on a single split

Requirements
------------
    pip install ultralytics opencv-python numpy
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR       = os.path.abspath(os.path.dirname(__file__))
DATA_YAML      = os.path.join(BASE_DIR, "data.yaml")

# Fine-tuned YOLO weights (latest training run)
YOLO_WEIGHTS   = os.path.join(BASE_DIR, "runs", "detect", "train28", "weights", "best.pt")
# Fallback to base weights if fine-tuned model is missing
YOLO_FALLBACK  = os.path.join(BASE_DIR, "yolov8n.pt")

CROPS_DIR      = os.path.join(BASE_DIR, "runs", "detect", "predict", "crops")
CLASS_NAMES    = {0: "authentic", 1: "counterfeit"}

CONFIDENCE     = 0.25        # minimum detection confidence
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_yolo_model():
    """Load the best available YOLOv8 model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed. Run:  pip install ultralytics")
        sys.exit(1)

    weights = YOLO_WEIGHTS if os.path.isfile(YOLO_WEIGHTS) else YOLO_FALLBACK
    if not os.path.isfile(weights):
        print(f"ERROR: No YOLO weights found at {YOLO_WEIGHTS} or {YOLO_FALLBACK}")
        sys.exit(1)

    print(f"  Loading YOLO model from: {weights}")
    model = YOLO(weights)
    return model


def detect_and_crop(model, image_path, output_dir, conf=CONFIDENCE):
    """
    Run YOLOv8 detection on a single image.
    Crop each detected object and save to output_dir/<class_name>/.

    Returns
    -------
    list[dict]  – one dict per detection with keys:
        class_id, class_name, confidence, bbox, crop_path
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    results = model.predict(source=image_path, conf=conf, verbose=False)
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for i, box in enumerate(boxes):
            cls_id   = int(box.cls[0])
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Clamp to image bounds
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

            # Save crop
            cls_dir = os.path.join(output_dir, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            stem = Path(image_path).stem
            crop_name = f"{stem}_crop{i}.jpg"
            crop_path = os.path.join(cls_dir, crop_name)
            cv2.imwrite(crop_path, crop)

            detections.append({
                "class_id":    cls_id,
                "class_name":  cls_name,
                "confidence":  conf_val,
                "bbox":        (x1, y1, x2, y2),
                "crop_path":   crop_path,
            })

    return detections


def process_split(model, split_name, base_dir=BASE_DIR, output_dir=CROPS_DIR):
    """Run detection on all images in a dataset split."""
    img_dir = os.path.join(base_dir, split_name, "images")
    if not os.path.isdir(img_dir):
        print(f"  WARNING: {img_dir} not found — skipping {split_name}")
        return

    image_files = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith(IMG_EXTENSIONS)
    ])
    print(f"\n  [{split_name.upper()}] Processing {len(image_files)} images ...")

    total_detections = 0
    for idx, img_path in enumerate(image_files):
        dets = detect_and_crop(model, img_path, output_dir)
        total_detections += len(dets)
        if (idx + 1) % 200 == 0:
            print(f"    ... {idx + 1}/{len(image_files)} images processed")

    print(f"  [{split_name.upper()}] Done — {total_detections} objects cropped")


def show_sample_detections(model, split_name="test", n_samples=4):
    """Visualise bounding boxes on a few sample images."""
    img_dir = os.path.join(BASE_DIR, split_name, "images")
    if not os.path.isdir(img_dir):
        return

    files = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith(IMG_EXTENSIONS)
    ])[:n_samples]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, min(n_samples, len(files)), figsize=(5 * len(files), 5))
    if len(files) == 1:
        axes = [axes]

    for ax, img_path in zip(axes, files):
        img = cv2.imread(img_path)
        results = model.predict(source=img_path, conf=CONFIDENCE, verbose=False)

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
                label = f"{CLASS_NAMES.get(cls_id, '?')} {conf_val:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(Path(img_path).name, fontsize=9)
        ax.axis("off")

    plt.suptitle("YOLOv8 Medicine Detection — Sample Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "results", "yolo_sample_detections.png"), dpi=150)
    plt.show()
    print("  Sample detections saved to results/yolo_sample_detections.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Detection & Cropping")
    parser.add_argument("--split", type=str, default=None,
                        help="Process a single split (train/valid/test). Default: all")
    parser.add_argument("--visualize", action="store_true",
                        help="Show sample detection visualisations")
    args = parser.parse_args()

    print("=" * 70)
    print("  YOLOv8 OBJECT DETECTION & CROPPING")
    print("  Counterfeit Medicine Detection Pipeline — Stage 1")
    print("=" * 70)

    # Check if crops already exist
    if os.path.isdir(CROPS_DIR):
        n_auth = len(os.listdir(os.path.join(CROPS_DIR, "authentic"))) \
            if os.path.isdir(os.path.join(CROPS_DIR, "authentic")) else 0
        n_fake = len(os.listdir(os.path.join(CROPS_DIR, "counterfeit"))) \
            if os.path.isdir(os.path.join(CROPS_DIR, "counterfeit")) else 0
        if n_auth + n_fake > 0:
            print(f"\n  Crops already exist: {n_auth} authentic, {n_fake} counterfeit")
            print("  Delete the crops directory to re-run detection.")
            if not args.visualize:
                return

    model = load_yolo_model()

    if not args.visualize:
        splits = [args.split] if args.split else ["train", "valid", "test"]
        for split in splits:
            process_split(model, split)

        # Summary
        print("\n" + "=" * 70)
        for cls_name in CLASS_NAMES.values():
            cls_dir = os.path.join(CROPS_DIR, cls_name)
            if os.path.isdir(cls_dir):
                n = len([f for f in os.listdir(cls_dir) if f.lower().endswith(IMG_EXTENSIONS)])
                print(f"  {cls_name}: {n} cropped images")
        print("=" * 70)

    if args.visualize:
        show_sample_detections(model)


if __name__ == "__main__":
    main()
