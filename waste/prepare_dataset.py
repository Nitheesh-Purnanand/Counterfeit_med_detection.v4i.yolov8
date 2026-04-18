"""
prepare_dataset.py -- Convert YOLOv8 object detection dataset to classification format
Based on: IOP Conf. Ser.: Earth Environ. Sci. 1529 (2025) 012032

Converts the 77-class Roboflow e-waste dataset (bounding box annotations)
into the paper's 10-class classification dataset by:
1. Mapping 77 Roboflow classes -> 10 paper classes
2. Cropping individual objects from bounding boxes
3. Resizing all crops to 180x180 pixels
4. Balancing classes to 240 images/class (train) via under/oversampling + augmentation
"""

import os
import sys
import random
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    print("Installing Pillow...")
    os.system(f"{sys.executable} -m pip install Pillow -q")
    from PIL import Image, ImageEnhance, ImageFilter

# ===================================================================
# CONFIGURATION
# ===================================================================

# Source dataset (YOLOv8 format)
YOLO_DATASET_DIR = os.path.join(
    os.path.dirname(__file__),
    "E-Waste Dataset.v44-fix-annotations-of-some-bar-phones-incorrectly-labelled-as-smartphones.yolov8"
)

# Output classification dataset
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dataset")

# Paper specifications
IMG_SIZE = 180          # Paper Section 2.3: 180x180 pixels
TRAIN_PER_CLASS = 240   # Paper: 2400 total / 10 classes = 240 per class
VAL_PER_CLASS = 60      # ~300 total validation / 10 classes
TEST_PER_CLASS = 30     # ~300 total test / 10 classes

# Minimum crop size (skip tiny bounding boxes)
MIN_CROP_PX = 20

# Random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ===================================================================
# 77 -> 10 CLASS MAPPING (Verified against paper Section 2.2)
# ===================================================================

# Roboflow class names (from data.yaml, 0-indexed)
ROBOFLOW_NAMES = [
    'Air-Conditioner', 'Bar-Phone', 'Battery', 'Blood-Pressure-Monitor',
    'Boiler', 'CRT-Monitor', 'CRT-TV', 'Calculator', 'Camera',
    'Ceiling-Fan', 'Christmas-Lights', 'Clothes-Iron', 'Coffee-Machine',
    'Compact-Fluorescent-Lamps', 'Computer-Keyboard', 'Computer-Mouse',
    'Cooled-Dispenser', 'Cooling-Display', 'Dehumidifier', 'Desktop-PC',
    'Digital-Oscilloscope', 'Dishwasher', 'Drone', 'Electric-Bicycle',
    'Electric-Guitar', 'Electrocardiograph-Machine', 'Electronic-Keyboard',
    'Exhaust-Fan', 'Flashlight', 'Flat-Panel-Monitor', 'Flat-Panel-TV',
    'Floor-Fan', 'Freezer', 'Glucose-Meter', 'HDD', 'Hair-Dryer',
    'Headphone', 'LED-Bulb', 'Laptop', 'Microwave', 'Music-Player',
    'Neon-Sign', 'Network-Switch', 'Non-Cooled-Dispenser', 'Oven', 'PCB',
    'Patient-Monitoring-System', 'Photovoltaic-Panel', 'PlayStation-5',
    'Power-Adapter', 'Printer', 'Projector', 'Pulse-Oximeter', 'Range-Hood',
    'Refrigerator', 'Rotary-Mower', 'Router', 'SSD', 'Server',
    'Smart-Watch', 'Smartphone', 'Smoke-Detector', 'Soldering-Iron',
    'Speaker', 'Stove', 'Straight-Tube-Fluorescent-Lamp', 'Street-Lamp',
    'TV-Remote-Control', 'Table-Lamp', 'Tablet', 'Telephone-Set', 'Toaster',
    'Tumble-Dryer', 'USB-Flash-Drive', 'Vacuum-Cleaner', 'Washing-Machine',
    'Xbox-Series-X'
]

# Paper's 10 class names (Section 2.2)
PAPER_CLASSES = [
    'Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse',
    'PCB', 'Player', 'Printer', 'Television', 'WashingMachine'
]

# Mapping: Roboflow class ID -> Paper class name
# Only IDs that clearly belong to the paper's categories are included
ROBOFLOW_TO_PAPER = {
    2:  'Battery',          # Battery
    14: 'Keyboard',         # Computer-Keyboard
    26: 'Keyboard',         # Electronic-Keyboard
    39: 'Microwave',        # Microwave
    1:  'Mobile',           # Bar-Phone
    60: 'Mobile',           # Smartphone
    15: 'Mouse',            # Computer-Mouse
    45: 'PCB',              # PCB
    40: 'Player',           # Music-Player
    48: 'Player',           # PlayStation-5
    76: 'Player',           # Xbox-Series-X
    50: 'Printer',          # Printer
    51: 'Printer',          # Projector (included -- Printer alone has only 9 images)
    6:  'Television',       # CRT-TV
    30: 'Television',       # Flat-Panel-TV
    5:  'Television',       # CRT-Monitor
    29: 'Television',       # Flat-Panel-Monitor
    75: 'WashingMachine',   # Washing-Machine
    72: 'WashingMachine',   # Tumble-Dryer
}


# ===================================================================
# STEP 1: CROP OBJECTS FROM YOLO BOUNDING BOXES
# ===================================================================

def parse_yolo_label(label_path, img_w, img_h):
    """Parse a YOLO format label file and return list of (class_id, x1, y1, x2, y2)."""
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            
            x1 = max(0, int(x_center - w / 2))
            y1 = max(0, int(y_center - h / 2))
            x2 = min(img_w, int(x_center + w / 2))
            y2 = min(img_h, int(y_center + h / 2))
            
            # Skip tiny crops
            if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
                continue
            
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def crop_and_save_objects(split_name, yolo_dir, output_dir):
    """Crop objects from images using YOLO bounding boxes and save to class folders."""
    images_dir = os.path.join(yolo_dir, split_name, 'images')
    labels_dir = os.path.join(yolo_dir, split_name, 'labels')
    
    if not os.path.isdir(images_dir):
        print(f"  WARNING: {images_dir} not found, skipping.")
        return {}
    
    crop_counts = defaultdict(int)
    skipped = 0
    errors = 0
    
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"  Processing {len(image_files)} images from {split_name}...")
    
    for i, img_file in enumerate(image_files):
        if (i + 1) % 1000 == 0:
            print(f"    Processed {i + 1}/{len(image_files)} images...")
        
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        if not os.path.exists(label_path):
            skipped += 1
            continue
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_w, img_h = img.size
        except Exception as e:
            errors += 1
            continue
        
        boxes = parse_yolo_label(label_path, img_w, img_h)
        
        for cls_id, x1, y1, x2, y2 in boxes:
            if cls_id not in ROBOFLOW_TO_PAPER:
                continue  # Not one of our 10 classes
            
            paper_class = ROBOFLOW_TO_PAPER[cls_id]
            
            try:
                crop = img.crop((x1, y1, x2, y2))
                crop = crop.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                
                # Save to output directory
                class_dir = os.path.join(output_dir, split_name, paper_class)
                os.makedirs(class_dir, exist_ok=True)
                
                crop_idx = crop_counts[paper_class]
                save_name = f"{paper_class}_{split_name}_{crop_idx:05d}.jpg"
                crop.save(os.path.join(class_dir, save_name), 'JPEG', quality=95)
                
                crop_counts[paper_class] += 1
            except Exception as e:
                errors += 1
                continue
    
    print(f"  Done: {sum(crop_counts.values())} crops saved, {skipped} no-label, {errors} errors")
    return dict(crop_counts)


# ===================================================================
# STEP 2: BALANCE DATASET (UNDERSAMPLE + AUGMENT)
# ===================================================================

def augment_image(img):
    """Apply random augmentation to a PIL image (offline augmentation for minority classes)."""
    # Random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Random vertical flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Random rotation (+-15 degrees)
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, fillcolor=(0, 0, 0))
    
    # Random brightness adjustment
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Random contrast adjustment
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Random slight blur or sharpen
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
    
    return img


def balance_split(split_dir, target_per_class):
    """Balance a split directory so each class has exactly target_per_class images."""
    print(f"\n  Balancing {split_dir} to {target_per_class} per class...")
    
    for class_name in PAPER_CLASSES:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"    WARNING: {class_name} directory missing! Creating empty directory.")
            os.makedirs(class_dir, exist_ok=True)
            continue
        
        images = [f for f in os.listdir(class_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        current_count = len(images)
        
        if current_count == 0:
            print(f"    WARNING: {class_name} has 0 images! Cannot balance.")
            continue
        
        if current_count >= target_per_class:
            # UNDERSAMPLE: randomly remove excess images
            random.shuffle(images)
            to_remove = images[target_per_class:]
            for f in to_remove:
                os.remove(os.path.join(class_dir, f))
            print(f"    {class_name}: {current_count} -> {target_per_class} (removed {len(to_remove)})")
        else:
            # OVERSAMPLE: augment existing images to reach target
            needed = target_per_class - current_count
            aug_count = 0
            
            while aug_count < needed:
                # Pick a random source image
                source_file = random.choice(images)
                source_path = os.path.join(class_dir, source_file)
                
                try:
                    img = Image.open(source_path).convert('RGB')
                    aug_img = augment_image(img)
                    
                    save_name = f"{class_name}_aug_{aug_count:05d}.jpg"
                    aug_img.save(os.path.join(class_dir, save_name), 'JPEG', quality=95)
                    aug_count += 1
                except Exception:
                    continue
            
            print(f"    {class_name}: {current_count} -> {target_per_class} (augmented +{needed})")


# ===================================================================
# STEP 3: GENERATE DATASET STATISTICS REPORT
# ===================================================================

def print_dataset_stats(output_dir):
    """Print a summary of the final dataset."""
    print("\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)
    
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(output_dir, split)
        if not os.path.isdir(split_dir):
            continue
        
        print(f"\n{'-' * 40}")
        print(f"Split: {split.upper()}")
        print(f"{'-' * 40}")
        
        total = 0
        for class_name in PAPER_CLASSES:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                count = len([f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            else:
                count = 0
            total += count
            status = "[OK]" if count > 0 else "[X] MISSING"
            print(f"  {class_name:20s}: {count:5d} images  {status}")
        
        print(f"  {'TOTAL':20s}: {total:5d} images")


def balance_split_full(split_dir, min_per_class=100):
    """Balance by only augmenting classes below min_per_class. No undersampling."""
    print(f"\n  Balancing (FULL mode) {split_dir}...")
    print(f"  (Augmenting classes with fewer than {min_per_class} images)")
    
    for class_name in PAPER_CLASSES:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"    WARNING: {class_name} directory missing!")
            os.makedirs(class_dir, exist_ok=True)
            continue
        
        images = [f for f in os.listdir(class_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        current_count = len(images)
        
        if current_count >= min_per_class:
            print(f"    {class_name}: {current_count} images (OK)")
        elif current_count > 0:
            # Augment up to min_per_class
            needed = min_per_class - current_count
            aug_count = 0
            while aug_count < needed:
                source_file = random.choice(images)
                source_path = os.path.join(class_dir, source_file)
                try:
                    img = Image.open(source_path).convert('RGB')
                    aug_img = augment_image(img)
                    save_name = f"{class_name}_aug_{aug_count:05d}.jpg"
                    aug_img.save(os.path.join(class_dir, save_name), 'JPEG', quality=95)
                    aug_count += 1
                except Exception:
                    continue
            print(f"    {class_name}: {current_count} -> {min_per_class} (augmented +{needed})")
        else:
            print(f"    WARNING: {class_name} has 0 images!")


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 60)
    print("E-WASTE DATASET PREPARATION")
    print("YOLO v8 Object Detection -> Classification Format")
    print("Based on: IOP Conf. Ser.: Earth Environ. Sci. 1529 (2025)")
    print("=" * 60)
    
    # Verify source dataset exists
    if not os.path.isdir(YOLO_DATASET_DIR):
        print(f"ERROR: Source dataset not found at:\n  {YOLO_DATASET_DIR}")
        sys.exit(1)
    
    # ================================================================
    # PART A: Create CAPPED dataset (240/class) -- for paper baseline
    # ================================================================
    print("\n" + "=" * 60)
    print("PART A: Creating CAPPED dataset (240/class)")
    print("=" * 60)
    
    if os.path.exists(OUTPUT_DIR):
        print(f"\nRemoving existing: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "-" * 60)
    print("STEP 1: Cropping objects from YOLO bounding boxes")
    print("-" * 60)
    
    all_counts = {}
    for split in ['train', 'valid', 'test']:
        print(f"\n[{split.upper()}]")
        counts = crop_and_save_objects(split, YOLO_DATASET_DIR, OUTPUT_DIR)
        all_counts[split] = counts
        for cls, cnt in sorted(counts.items()):
            print(f"    {cls:20s}: {cnt:5d} crops")
    
    print("\n" + "-" * 60)
    print("STEP 2: Balancing (capped at 240/60/30)")
    print("-" * 60)
    
    balance_split(os.path.join(OUTPUT_DIR, 'train'), TRAIN_PER_CLASS)
    balance_split(os.path.join(OUTPUT_DIR, 'valid'), VAL_PER_CLASS)
    balance_split(os.path.join(OUTPUT_DIR, 'test'), TEST_PER_CLASS)
    
    print_dataset_stats(OUTPUT_DIR)
    
    # ================================================================
    # PART B: Create FULL dataset (all crops, only augment minority)
    # ================================================================
    FULL_DIR = os.path.join(os.path.dirname(__file__), "dataset_full")
    
    print("\n\n" + "=" * 60)
    print("PART B: Creating FULL dataset (all available crops)")
    print("=" * 60)
    
    if os.path.exists(FULL_DIR):
        print(f"\nRemoving existing: {FULL_DIR}")
        shutil.rmtree(FULL_DIR)
    os.makedirs(FULL_DIR, exist_ok=True)
    
    print("\n" + "-" * 60)
    print("STEP 1: Cropping ALL objects")
    print("-" * 60)
    
    for split in ['train', 'valid', 'test']:
        print(f"\n[{split.upper()}]")
        counts = crop_and_save_objects(split, YOLO_DATASET_DIR, FULL_DIR)
        for cls, cnt in sorted(counts.items()):
            print(f"    {cls:20s}: {cnt:5d} crops")
    
    print("\n" + "-" * 60)
    print("STEP 2: Augmenting minority classes only")
    print("-" * 60)
    
    # For train: augment classes below 300 (roughly the median)
    balance_split_full(os.path.join(FULL_DIR, 'train'), min_per_class=300)
    # For valid/test: augment classes below 60/30
    balance_split_full(os.path.join(FULL_DIR, 'valid'), min_per_class=60)
    balance_split_full(os.path.join(FULL_DIR, 'test'), min_per_class=30)
    
    print_dataset_stats(FULL_DIR)
    
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print(f"  Capped dataset: {OUTPUT_DIR}")
    print(f"  Full dataset:   {FULL_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()

