"""
=============================================================================
Counterfeit Medicine Detection Using ML Classifiers
WITH DATA AUGMENTATION FOR CLASS IMBALANCE

Base Paper: "Analyzing the Counterfeit Medicines Based on Classification
            Using Machine Learning Techniques"
            - Binitha S. Thomson and W. Rose Varuna (Springer, 2024)

Our Contribution:
  - Image-level augmentation of minority class (counterfeit) to balance data
  - SMOTE feature-level oversampling as additional balancing strategy
  - XGBoost classifier (added to paper's RF, NB, KNN, SVM)
  - Comprehensive evaluation with focus on counterfeit recall/F1

Pipeline:
  1. Load cropped medicine images (from YOLOv8 detection)
  2. Augment the minority (counterfeit) class with image transforms
  3. Merge & stratified 80/20 train/test split
  4. Extract features (Color Histogram, LBP, GLCM, Hu Moments, HOG)
  5. Apply SMOTE oversampling on features (training set only)
  6. Train & evaluate 5 ML classifiers with class-weighted losses
  7. Compare results with focus on counterfeit detection accuracy
  8. Generate visualizations and reports
=============================================================================
"""

import os
import sys
import warnings
import time
import random
import pickle
import numpy as np
import cv2
from collections import Counter
from tqdm import tqdm

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score,
                             recall_score, f1_score)
from xgboost import XGBClassifier

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR
CROPS_DIR = os.path.join(BASE_DIR, "runs", "detect", "predict", "crops")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
AUG_DIR = os.path.join(BASE_DIR, "augmented_crops")

IMG_SIZE = 128          # Resize all images to 128x128
TEST_RATIO = 0.20       # 80/20 split as per paper
RANDOM_STATE = 42
SEED = 42

# Augmentation settings
TARGET_RATIO = 1.0      # Target ratio: counterfeit/authentic (1.0 = balanced)

random.seed(SEED)
np.random.seed(SEED)


# ============================================================================
# GPU DETECTION
# ============================================================================
def detect_gpu():
    """Detect available GPU for XGBoost acceleration."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {name} ({mem:.1f} GB) — XGBoost will use CUDA")
            return 'cuda'
    except Exception:
        pass

    try:
        import xgboost as xgb
        test_clf = xgb.XGBClassifier(device='cuda', n_estimators=1, verbosity=0)
        test_clf.fit(np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([0, 1]))
        print("  GPU detected via XGBoost. Using CUDA.")
        return 'cuda'
    except Exception:
        pass

    print("  No CUDA GPU available. Using CPU.")
    return 'cpu'


# ============================================================================
# SECTION 1: DATASET EXPLORATION
# ============================================================================
def explore_dataset():
    """Count images & annotations per split, show class distribution."""
    print("\n" + "=" * 70)
    print("  SECTION 1: DATASET EXPLORATION")
    print("=" * 70)

    class_dist = {}
    for split in ['train', 'valid', 'test']:
        lbl_dir = os.path.join(BASE_DIR, split, 'labels')
        if not os.path.isdir(lbl_dir):
            print(f"  WARNING: {lbl_dir} not found")
            continue

        img_dir = os.path.join(BASE_DIR, split, 'images')
        n_img = 0
        if os.path.isdir(img_dir):
            n_img = len([f for f in os.listdir(img_dir)
                         if f.endswith(('.jpg', '.png'))])

        counter = Counter()
        for lbl_file in os.listdir(lbl_dir):
            if not lbl_file.endswith('.txt'):
                continue
            with open(os.path.join(lbl_dir, lbl_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        counter[parts[0]] += 1

        class_dist[split] = counter
        auth = counter.get('0', 0)
        fake = counter.get('1', 0)
        print(f"  {split.upper():>6} -- {n_img} images | "
              f"Authentic: {auth:>5} | Counterfeit: {fake:>4} | Total: {auth+fake}")

    total_auth = sum(d.get('0', 0) for d in class_dist.values())
    total_fake = sum(d.get('1', 0) for d in class_dist.values())
    ratio = total_auth / max(total_fake, 1)
    print(f"\n  TOTAL -- Authentic: {total_auth} | Counterfeit: {total_fake}")
    print(f"  Class ratio: {ratio:.1f}:1 (authentic : counterfeit)")

    if ratio > 2.0:
        print(f"\n  >>> SEVERE IMBALANCE DETECTED! Augmentation is CRITICAL. <<<")

    return class_dist


# ============================================================================
# SECTION 2: IMAGE-LEVEL AUGMENTATION FOR MINORITY CLASS
# ============================================================================
def augment_image(img):
    """Apply random augmentation to a single image (BGR format)."""
    aug = img.copy()

    if random.random() > 0.5:
        aug = cv2.flip(aug, 1)

    if random.random() > 0.5:
        aug = cv2.flip(aug, 0)

    if random.random() > 0.3:
        angle = random.uniform(-30, 30)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    if random.random() > 0.3:
        alpha = random.uniform(0.7, 1.3)
        beta = random.randint(-30, 30)
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    if random.random() > 0.7:
        ksize = random.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)

    if random.random() > 0.7:
        noise = np.random.normal(0, 10, aug.shape).astype(np.int16)
        aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if random.random() > 0.5:
        h, w = aug.shape[:2]
        scale = random.uniform(0.8, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        y_off = (h - new_h) // 2
        x_off = (w - new_w) // 2
        aug = cv2.resize(aug[y_off:y_off+new_h, x_off:x_off+new_w], (w, h))

    return aug


def augment_minority_class(crops_dir, aug_dir, target_ratio=TARGET_RATIO):
    """
    Augment the minority class (counterfeit) images to balance the dataset.
    Creates augmented copies in aug_dir while keeping originals.
    """
    print("\n" + "=" * 70)
    print("  SECTION 2: IMAGE-LEVEL AUGMENTATION")
    print("=" * 70)

    auth_dir = os.path.join(crops_dir, 'authentic')
    fake_dir = os.path.join(crops_dir, 'counterfeit')

    if not os.path.isdir(auth_dir) or not os.path.isdir(fake_dir):
        print(f"  ERROR: Crop directories not found at {crops_dir}")
        print(f"  Run YOLOv8 prediction with save_crop=True first.")
        sys.exit(1)

    exts = ('.jpg', '.jpeg', '.png')
    auth_files = [f for f in os.listdir(auth_dir) if f.lower().endswith(exts)]
    fake_files = [f for f in os.listdir(fake_dir) if f.lower().endswith(exts)]
    n_auth, n_fake = len(auth_files), len(fake_files)

    print(f"\n  Original counts:")
    print(f"    Authentic:   {n_auth}")
    print(f"    Counterfeit: {n_fake}")
    print(f"    Ratio: {n_auth / max(n_fake, 1):.1f}:1")

    target_fake = int(n_auth * target_ratio)
    n_aug_needed = max(0, target_fake - n_fake)

    if n_aug_needed == 0:
        print("  Classes already balanced. No augmentation needed.")
        return 0

    print(f"\n  Target counterfeit count: {target_fake}")
    print(f"  Augmented images needed:  {n_aug_needed}")

    aug_fake_dir = os.path.join(aug_dir, 'counterfeit')
    os.makedirs(aug_fake_dir, exist_ok=True)
    os.makedirs(os.path.join(aug_dir, 'authentic'), exist_ok=True)

    aug_count = 0
    aug_round = 0

    print(f"\n  Generating augmented counterfeit images...")
    with tqdm(total=n_aug_needed, desc="  Augmenting") as pbar:
        while aug_count < n_aug_needed:
            aug_round += 1
            random.shuffle(fake_files)
            for fake_file in fake_files:
                if aug_count >= n_aug_needed:
                    break
                img = cv2.imread(os.path.join(fake_dir, fake_file))
                if img is None:
                    continue

                aug_img = augment_image(img)
                base = os.path.splitext(fake_file)[0]
                save_path = os.path.join(aug_fake_dir,
                                         f"{base}_aug{aug_round}_{aug_count}.jpg")
                cv2.imwrite(save_path, aug_img)
                aug_count += 1
                pbar.update(1)

    print(f"\n  Augmentation complete!")
    print(f"    Original counterfeit:  {n_fake}")
    print(f"    Augmented counterfeit: {aug_count}")
    print(f"    Total counterfeit:     {n_fake + aug_count}")
    print(f"    New ratio: {n_auth / max(n_fake + aug_count, 1):.2f}:1")
    return aug_count


# ============================================================================
# SECTION 3: DATA LOADING (with augmented images)
# ============================================================================
def load_all_image_paths(crops_dir, aug_dir):
    """Load image paths from original crops + augmented directory."""
    print("\n" + "=" * 70)
    print("  SECTION 3: DATA LOADING")
    print("=" * 70)

    image_paths = []
    labels = []
    exts = ('.jpg', '.jpeg', '.png', '.bmp')

    for class_name, label in [('authentic', 0), ('counterfeit', 1)]:
        class_dir = os.path.join(crops_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  WARNING: {class_dir} not found")
            continue
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(exts)]
        for f in files:
            image_paths.append(os.path.join(class_dir, f))
            labels.append(label)
        print(f"  Original {class_name}: {len(files)} images")

    # Load augmented counterfeit images
    aug_fake_dir = os.path.join(aug_dir, 'counterfeit')
    if os.path.isdir(aug_fake_dir):
        aug_files = [f for f in os.listdir(aug_fake_dir)
                     if f.lower().endswith(exts)]
        for f in aug_files:
            image_paths.append(os.path.join(aug_fake_dir, f))
            labels.append(1)
        print(f"  Augmented counterfeit: {len(aug_files)} images")

    labels = np.array(labels)
    n_auth = int(np.sum(labels == 0))
    n_fake = int(np.sum(labels == 1))
    print(f"\n  Total images: {len(image_paths)}")
    print(f"    Authentic (0):   {n_auth}")
    print(f"    Counterfeit (1): {n_fake}")
    print(f"    Ratio: {n_auth / max(n_fake, 1):.2f}:1")

    return image_paths, labels


# ============================================================================
# SECTION 4: FEATURE EXTRACTION  (sequential — safe on Windows)
# ============================================================================
def extract_color_histogram(img_bgr):
    """Extract HSV color histogram (8x8x8 bins)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()   # 512


def extract_lbp_features(img_gray):
    """Extract Local Binary Pattern histogram."""
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                           range=(0, n_bins), density=True)
    return hist   # 26


def extract_glcm_features(img_gray):
    """Extract GLCM texture features."""
    img_q = (img_gray // 4).astype(np.uint8)   # quantize to 64 levels
    glcm = graycomatrix(img_q, distances=[1, 3],
                        angles=[0, np.pi / 4, np.pi / 2],
                        levels=64, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    return np.array([graycoprops(glcm, p).mean() for p in props])   # 5


def extract_hu_moments(img_gray):
    """Extract Hu Moments (shape-invariant)."""
    hu = cv2.HuMoments(cv2.moments(img_gray)).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)   # 7


def extract_hog_features(img_gray):
    """Extract HOG features."""
    return hog(img_gray, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), feature_vector=True)


def extract_all_features(img_path):
    """Extract complete feature vector from a single image."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    return np.concatenate([
        extract_color_histogram(img_bgr),     # 512
        extract_lbp_features(img_gray),       # 26
        extract_glcm_features(img_gray),      # 5
        extract_hu_moments(img_gray),         # 7
        extract_hog_features(img_gray),       # ~1764
    ])


def extract_features_batch(image_paths, desc="Extracting"):
    """Extract features sequentially (safe, no multiprocessing issues)."""
    features = []
    valid_idx = []
    failed = 0

    for i, path in enumerate(tqdm(image_paths, desc=desc)):
        feat = extract_all_features(path)
        if feat is not None:
            features.append(feat)
            valid_idx.append(i)
        else:
            failed += 1

    if failed > 0:
        print(f"  WARNING: {failed} images failed to load")

    return np.array(features, dtype=np.float32), valid_idx


# ============================================================================
# SECTION 5: SMOTE / OVERSAMPLING
# ============================================================================
def apply_smote(X_train, y_train):
    """Apply oversampling only if classes are still imbalanced."""
    print("\n  Checking class balance...")
    counts = Counter(y_train)
    minority = min(counts.values())
    majority = max(counts.values())
    ratio = majority / max(minority, 1)

    print(f"    Class distribution: {dict(counts)}")
    print(f"    Imbalance ratio: {ratio:.2f}:1")

    if ratio < 1.2:
        print(f"    Already balanced (ratio < 1.2). Skipping oversampling.")
        return X_train, y_train

    print(f"    Imbalanced → applying RandomOverSampler...")
    try:
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=RANDOM_STATE)
        X_res, y_res = ros.fit_resample(X_train, y_train)
        print(f"    After oversampling: {dict(Counter(y_res))}")
        return X_res.astype(np.float32), y_res
    except (ImportError, MemoryError) as e:
        print(f"    Oversampling failed ({e}). Continuing with original data.")
        return X_train, y_train


# ============================================================================
# SECTION 6: ML MODEL TRAINING & EVALUATION
# ============================================================================
def build_classifiers(scale_pos_weight, gpu_device):
    """Return an OrderedDict of classifiers."""
    return {
        'Random Forest (RF)': RandomForestClassifier(
            n_estimators=200, criterion='gini', class_weight='balanced',
            max_depth=None, min_samples_split=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        'Naive Bayes (NB)': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1),
        'SVM': SVC(
            kernel='rbf', class_weight='balanced', C=10, gamma='scale',
            random_state=RANDOM_STATE,
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric='logloss',
            tree_method='hist', device=gpu_device,
            random_state=RANDOM_STATE, n_jobs=2,
        ),
    }


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       class_names, gpu_device='cpu'):
    """Train all classifiers, evaluate, return results dict."""
    print("\n" + "=" * 70)
    print("  SECTION 6: TRAINING ML CLASSIFIERS")
    print("=" * 70)

    spw = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
    classifiers = build_classifiers(spw, gpu_device)

    results = {}
    trained_models = {}

    for name, clf in classifiers.items():
        print(f"\n  Training {name}...")
        t0 = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        elapsed = time.time() - t0

        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, average='weighted') * 100
        rec = recall_score(y_test, y_pred, average='weighted') * 100
        f1 = f1_score(y_test, y_pred, average='weighted') * 100

        if np.sum(y_test == 1) > 0:
            fake_recall = recall_score(y_test, y_pred, pos_label=1) * 100
            fake_prec = precision_score(y_test, y_pred, pos_label=1) * 100
            fake_f1 = f1_score(y_test, y_pred, pos_label=1) * 100
        else:
            fake_recall = fake_prec = fake_f1 = 0.0

        report = classification_report(
            y_test, y_pred, target_names=class_names, digits=4)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'fake_recall': fake_recall, 'fake_precision': fake_prec,
            'fake_f1': fake_f1, 'time': elapsed,
            'report': report, 'confusion_matrix': cm, 'y_pred': y_pred,
        }
        trained_models[name] = clf

        print(f"    Accuracy:           {acc:.2f}%")
        print(f"    Counterfeit Recall: {fake_recall:.2f}%")
        print(f"    Counterfeit Prec:   {fake_prec:.2f}%")
        print(f"    Counterfeit F1:     {fake_f1:.2f}%")
        print(f"    Time: {elapsed:.2f}s")

    return results, trained_models


# ============================================================================
# SECTION 7: RESULTS TABLE
# ============================================================================
def print_results_table(results, y_test):
    """Print results comparison table."""
    print("\n" + "=" * 70)
    print("  SECTION 7: RESULTS COMPARISON")
    print("=" * 70)

    n = len(y_test)
    header = (f"  {'Algorithm':<25} {'N':<8} {'Acc%':<9} "
              f"{'FakeRec%':<10} {'FakePrec%':<11} {'FakeF1%':<9}")
    print(f"\n{header}")
    print("  " + "-" * 80)
    for name, r in results.items():
        print(f"  {name:<25} {n:<8} {r['accuracy']:.2f}    "
              f"{r['fake_recall']:.2f}     {r['fake_precision']:.2f}      "
              f"{r['fake_f1']:.2f}")
    print("  " + "-" * 80)

    for name, r in results.items():
        print(f"\n  --- {name} Classification Report ---")
        print(r['report'])


# ============================================================================
# SECTION 8: VISUALIZATIONS
# ============================================================================
def generate_visualizations(results, y_test, y_train, output_dir, trained_models):
    """Generate all plots and save to output_dir."""
    print("\n" + "=" * 70)
    print("  SECTION 8: GENERATING VISUALIZATIONS")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    class_names = ['Authentic', 'Counterfeit']
    algo_names = list(results.keys())

    # ---- Plot 1: Accuracy Comparison ----
    fig, ax = plt.subplots(figsize=(12, 6))
    accs = [results[n]['accuracy'] for n in algo_names]
    colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c', '#9b59b6']
    bars = ax.bar(algo_names, accs, color=colors[:len(algo_names)],
                  edgecolor='black', linewidth=0.8)
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, a + 0.5,
                f'{a:.2f}%', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('ML Algorithm')
    ax.set_title('Accuracy Comparison (With Augmentation & SMOTE)',
                 fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison_augmented.png'),
                dpi=150)
    plt.close()
    print("  Saved: accuracy_comparison_augmented.png")

    # ---- Plot 2: Counterfeit-Specific Metrics ----
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(algo_names))
    w = 0.25
    recs = [results[n]['fake_recall'] for n in algo_names]
    precs = [results[n]['fake_precision'] for n in algo_names]
    f1s = [results[n]['fake_f1'] for n in algo_names]

    for offset, vals, label, color in [
        (-w, recs, 'Recall', '#e74c3c'),
        (0, precs, 'Precision', '#3498db'),
        (w, f1s, 'F1', '#2ecc71'),
    ]:
        b = ax.bar(x + offset, vals, w, label=f'Counterfeit {label}',
                   color=color, edgecolor='black')
        for bar in b:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}', ha='center', fontsize=9)

    ax.set_ylabel('Score (%)')
    ax.set_xlabel('ML Algorithm')
    ax.set_title('Counterfeit Detection Performance', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'counterfeit_metrics_augmented.png'),
                dpi=150)
    plt.close()
    print("  Saved: counterfeit_metrics_augmented.png")

    # ---- Plot 3: Confusion Matrices ----
    n_clf = len(results)
    n_cols = 3
    n_rows = (n_clf + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (name, r) in enumerate(results.items()):
        row, col = divmod(i, n_cols)
        cm = r['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[row][col], cbar=False, annot_kws={"size": 14})
        axes[row][col].set_title(
            f"{name}\nAcc: {r['accuracy']:.2f}% | "
            f"Fake Recall: {r['fake_recall']:.1f}%",
            fontweight='bold', fontsize=10)
        axes[row][col].set_ylabel('True')
        axes[row][col].set_xlabel('Predicted')

    for j in range(n_clf, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row][col].set_visible(False)

    plt.suptitle('Confusion Matrices (With Augmentation)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_augmented.png'),
                dpi=150)
    plt.close()
    print("  Saved: confusion_matrices_augmented.png")

    # ---- Plot 4: Class Distribution ----
    fig, ax = plt.subplots(figsize=(8, 5))
    train_counts = [int(np.sum(y_train == 0)), int(np.sum(y_train == 1))]
    test_counts = [int(np.sum(y_test == 0)), int(np.sum(y_test == 1))]
    x = np.arange(len(class_names))
    w2 = 0.35
    b1 = ax.bar(x - w2 / 2, train_counts, w2, label='Train',
                color='#27ae60', edgecolor='black')
    b2 = ax.bar(x + w2 / 2, test_counts, w2, label='Test',
                color='#2980b9', edgecolor='black')
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 10,
                    str(int(bar.get_height())), ha='center', fontweight='bold')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Class Distribution (After Augmentation + SMOTE)',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution_augmented.png'),
                dpi=150)
    plt.close()
    print("  Saved: class_distribution_augmented.png")

    # ---- Plot 5: Feature Importance (Random Forest) ----
    rf = trained_models.get('Random Forest (RF)')
    if rf is not None:
        imp = rf.feature_importances_
        n_color, n_lbp, n_glcm, n_hu = 512, 26, 5, 7
        n_hog = len(imp) - n_color - n_lbp - n_glcm - n_hu
        group_names = ['Color Histogram', 'LBP Texture', 'GLCM Texture',
                       'Hu Moments', 'HOG']
        sizes = [n_color, n_lbp, n_glcm, n_hu, n_hog]
        group_imp = []
        idx = 0
        for s in sizes:
            group_imp.append(np.sum(imp[idx:idx + s]))
            idx += s

        fig, ax = plt.subplots(figsize=(10, 6))
        c5 = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c']
        bars = ax.barh(group_names, group_imp, color=c5, edgecolor='black')
        for bar, v in zip(bars, group_imp):
            ax.text(bar.get_width() + 0.002,
                    bar.get_y() + bar.get_height() / 2,
                    f'{v:.4f}', va='center', fontweight='bold')
        ax.set_xlabel('Total Feature Importance')
        ax.set_title('Feature Group Importance (Random Forest)',
                     fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_augmented.png'),
                    dpi=150)
        plt.close()
        print("  Saved: feature_importance_augmented.png")


# ============================================================================
# INDEPENDENT TEST VERIFICATION
# ============================================================================
def verify_on_independent_test(trained_models, scaler, class_names,
                               ind_test_dir=None):
    """Verify best model on independent test images (if available)."""
    if ind_test_dir is None:
        ind_test_dir = os.path.join(BASE_DIR, "ind_test")

    if not os.path.isdir(ind_test_dir):
        print("\n  No independent test directory found. Skipping.")
        return

    print("\n" + "=" * 70)
    print("  INDEPENDENT TEST VERIFICATION")
    print("=" * 70)

    exts = ('.jpg', '.jpeg', '.png')
    test_images = [f for f in os.listdir(ind_test_dir)
                   if f.lower().endswith(exts)]

    if not test_images:
        print("  No test images found.")
        return

    print(f"  Found {len(test_images)} independent test images")

    for name, model in trained_models.items():
        print(f"\n  --- {name} Predictions ---")
        for img_file in test_images[:10]:
            feat = extract_all_features(os.path.join(ind_test_dir, img_file))
            if feat is not None:
                feat_sc = scaler.transform(feat.reshape(1, -1))
                pred = model.predict(feat_sc)[0]
                print(f"    {img_file}: {class_names[pred]}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    print("=" * 70)
    print("  COUNTERFEIT MEDICINE DETECTION")
    print("  WITH DATA AUGMENTATION FOR CLASS IMBALANCE")
    print("=" * 70)

    gpu_device = detect_gpu()
    cache_path = os.path.join(BASE_DIR, 'features_cache.npz')

    # ------------------------------------------------------------------
    # Try loading cached features first
    # ------------------------------------------------------------------
    X_train_scaled = X_test_scaled = y_train = y_test = scaler = None

    if os.path.exists(cache_path):
        print("\n  >>> LOADING CACHED FEATURES "
              "(delete features_cache.npz to re-extract)")
        try:
            cached = np.load(cache_path, allow_pickle=True)
            X_train_scaled = cached['X_train_scaled'].astype(np.float32)
            X_test_scaled = cached['X_test_scaled'].astype(np.float32)
            y_train = cached['y_train_valid']
            y_test = cached['y_test_valid']
            scaler = cached['scaler'].item()
            cached.close()                              # release file handle

            print(f"  Train: {X_train_scaled.shape}  "
                  f"(auth={np.sum(y_train==0)}, fake={np.sum(y_train==1)})")
            print(f"  Test:  {X_test_scaled.shape}  "
                  f"(auth={np.sum(y_test==0)}, fake={np.sum(y_test==1)})")
        except Exception as e:
            print(f"  Cache load failed ({e}). Will re-extract...")
            X_train_scaled = None
            try:
                os.remove(cache_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Full pipeline if cache miss
    # ------------------------------------------------------------------
    if X_train_scaled is None:
        explore_dataset()
        augment_minority_class(CROPS_DIR, AUG_DIR, target_ratio=TARGET_RATIO)

        image_paths, labels = load_all_image_paths(CROPS_DIR, AUG_DIR)

        print("\n  Stratified 80/20 train/test split...")
        paths_train, paths_test, y_train, y_test = train_test_split(
            image_paths, labels, test_size=TEST_RATIO,
            random_state=RANDOM_STATE, stratify=labels)

        print(f"  Train: {len(paths_train)} | Test: {len(paths_test)}")

        # Feature extraction (sequential — no multiprocessing)
        print("\n" + "=" * 70)
        print("  SECTION 4: FEATURE EXTRACTION")
        print("=" * 70)

        print("\n  Extracting TRAIN features...")
        X_train_raw, train_valid = extract_features_batch(
            paths_train, desc="  Train")
        y_train = y_train[train_valid]

        print("  Extracting TEST features...")
        X_test_raw, test_valid = extract_features_batch(
            paths_test, desc="  Test")
        y_test = y_test[test_valid]

        print(f"\n  Feature dim: {X_train_raw.shape[1]}")
        print(f"  Train: {X_train_raw.shape} | Test: {X_test_raw.shape}")

        # Scale
        print("  Scaling features (StandardScaler)...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)

        # Free raw arrays
        del X_train_raw, X_test_raw

        # Cache for next run
        print("  Caching features for future runs...")
        np.savez_compressed(
            cache_path,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train_valid=y_train,
            y_test_valid=y_test,
            scaler=np.array(scaler, dtype=object))
        print(f"  Cached → {cache_path}")

    # ------------------------------------------------------------------
    # SMOTE oversampling
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SECTION 5: SMOTE OVERSAMPLING")
    print("=" * 70)
    X_train_final, y_train_final = apply_smote(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # Train & evaluate
    # ------------------------------------------------------------------
    class_names = ['Authentic', 'Counterfeit']
    results, trained_models = train_and_evaluate(
        X_train_final, y_train_final,
        X_test_scaled, y_test,
        class_names, gpu_device=gpu_device)

    print_results_table(results, y_test)
    generate_visualizations(results, y_test, y_train_final,
                            OUTPUT_DIR, trained_models)
    verify_on_independent_test(trained_models, scaler, class_names)

    # ------------------------------------------------------------------
    # Save pipeline data (lightweight — no duplicate arrays)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SAVING PIPELINE DATA")
    print("=" * 70)

    save_data = {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_smote': X_train_final,
        'y_train_smote': y_train_final,
        'scaler': scaler,
        'results': results,
        'trained_models': trained_models,
        'class_names': class_names,
    }
    pkl_path = os.path.join(BASE_DIR, 'pipeline_data_augmented.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved: {pkl_path} ({os.path.getsize(pkl_path) / 1e6:.1f} MB)")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    best_acc = max(results, key=lambda k: results[k]['accuracy'])
    best_rec = max(results, key=lambda k: results[k]['fake_recall'])

    print(f"\n  Best Accuracy:    {best_acc}  →  {results[best_acc]['accuracy']:.2f}%")
    print(f"  Best Fake Recall: {best_rec}  →  {results[best_rec]['fake_recall']:.2f}%")

    print(f"\n  Ranking by COUNTERFEIT RECALL:")
    for rank, (name, r) in enumerate(
            sorted(results.items(),
                   key=lambda x: x[1]['fake_recall'], reverse=True), 1):
        print(f"    {rank}. {name}: Recall={r['fake_recall']:.2f}%, "
              f"Acc={r['accuracy']:.2f}%, F1={r['fake_f1']:.2f}%")

    print(f"\n  All results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    print("  Done!")
    print("=" * 70)


# ============================================================================
# ENTRY POINT — Windows requires this guard for safe execution
# ============================================================================
if __name__ == '__main__':
    main()
