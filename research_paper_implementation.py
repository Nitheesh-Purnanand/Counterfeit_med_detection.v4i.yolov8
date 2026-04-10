"""
=============================================================================
Research Paper Implementation:
"Analyzing the Counterfeit Medicines Based on Classification Using 
 Machine Learning Techniques"
 - Binitha S. Thomson and W. Rose Varuna

Methodology:
  1. Load cropped medicine images (authentic vs counterfeit)
  2. Merge all data and perform stratified 80/20 train/test split
  3. Extract image features (Color, Texture, Shape, HOG)
  4. Train 4 ML classifiers: RF, NB, KNN, SVM
  5. Compare accuracies and visualize results
=============================================================================
"""

import os
import warnings
import time
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
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = r"d:\research papers\jenil_sir_project\Counterfeit_med_detection.v4i.yolov8"
CROPS_DIR = os.path.join(BASE_DIR, "runs", "detect", "predict", "crops")
IMG_SIZE = 128          # Resize all images to 128x128
TEST_RATIO = 0.20       # 80/20 split as per paper
RANDOM_STATE = 42

print("=" * 70)
print(" COUNTERFEIT MEDICINE DETECTION USING ML CLASSIFIERS")
print(" Research Paper Implementation")
print("=" * 70)

# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================
print("\n[SECTION 1] Loading Data...")
print("-" * 50)

def load_image_paths(crops_dir):
    """Load all image paths and labels from crop directories."""
    image_paths = []
    labels = []
    
    for class_name in ['authentic', 'counterfeit']:
        class_dir = os.path.join(crops_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  WARNING: Directory not found: {class_dir}")
            continue
        
        files = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        label = 0 if class_name == 'authentic' else 1
        
        for f in files:
            image_paths.append(os.path.join(class_dir, f))
            labels.append(label)
        
        print(f"  {class_name}: {len(files)} images (label={label})")
    
    return image_paths, labels

image_paths, labels = load_image_paths(CROPS_DIR)
labels = np.array(labels)

print(f"\n  Total images: {len(image_paths)}")
print(f"  Class distribution: {dict(Counter(labels))}")
print(f"    0 (authentic):   {np.sum(labels == 0)}")
print(f"    1 (counterfeit): {np.sum(labels == 1)}")

# Stratified 80/20 split
paths_train, paths_test, y_train, y_test = train_test_split(
    image_paths, labels, test_size=TEST_RATIO, 
    random_state=RANDOM_STATE, stratify=labels
)

print(f"\n  Train set: {len(paths_train)} images")
print(f"    authentic:   {np.sum(y_train == 0)}, counterfeit: {np.sum(y_train == 1)}")
print(f"  Test set:  {len(paths_test)} images")
print(f"    authentic:   {np.sum(y_test == 0)}, counterfeit: {np.sum(y_test == 1)}")

# ============================================================================
# SECTION 2: FEATURE EXTRACTION
# ============================================================================
print("\n[SECTION 2] Extracting Features...")
print("-" * 50)

def extract_color_histogram(img_bgr):
    """Extract HSV color histogram (8x8x8 bins) - captures color patterns."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist  # 512 features

def extract_lbp_features(img_gray):
    """Extract Local Binary Pattern histogram - captures texture patterns."""
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    n_bins = n_points + 2  # 26 bins for uniform LBP
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist  # 26 features

def extract_glcm_features(img_gray):
    """Extract GLCM texture features - contrast, dissimilarity, etc."""
    # Quantize to fewer gray levels for GLCM
    img_q = (img_gray // 4).astype(np.uint8)  # 64 levels
    glcm = graycomatrix(img_q, distances=[1, 3], angles=[0, np.pi/4, np.pi/2],
                        levels=64, symmetric=True, normed=True)
    
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = []
    for prop in props:
        val = graycoprops(glcm, prop)
        features.append(val.mean())  # Average across distances and angles
    return np.array(features)  # 5 features

def extract_hu_moments(img_gray):
    """Extract Hu Moments - shape-invariant features."""
    moments = cv2.moments(img_gray)
    hu = cv2.HuMoments(moments).flatten()
    # Log transform for better numerical properties
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu  # 7 features

def extract_hog_features(img_gray):
    """Extract HOG features - captures edge/gradient patterns."""
    features = hog(img_gray, orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), feature_vector=True)
    return features  # Variable size based on image size

def extract_all_features(img_path, img_size=IMG_SIZE):
    """Extract complete feature vector from a single image."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    
    img_bgr = cv2.resize(img_bgr, (img_size, img_size))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Extract all feature groups
    color_hist = extract_color_histogram(img_bgr)       # 512
    lbp_feat = extract_lbp_features(img_gray)            # 26
    glcm_feat = extract_glcm_features(img_gray)          # 5
    hu_feat = extract_hu_moments(img_gray)               # 7
    hog_feat = extract_hog_features(img_gray)            # variable (~1764)
    
    # Concatenate all features
    features = np.concatenate([color_hist, lbp_feat, glcm_feat, hu_feat, hog_feat])
    return features

def extract_features_batch(image_paths, desc="Extracting"):
    """Extract features for a batch of images with progress bar."""
    features_list = []
    failed = 0
    valid_indices = []
    
    for i, path in enumerate(tqdm(image_paths, desc=desc)):
        feat = extract_all_features(path)
        if feat is not None:
            features_list.append(feat)
            valid_indices.append(i)
        else:
            failed += 1
    
    if failed > 0:
        print(f"  WARNING: {failed} images failed to load")
    
    return np.array(features_list), valid_indices

# Extract features for train and test sets
print("\n  Extracting TRAIN features...")
X_train, train_valid = extract_features_batch(paths_train, desc="  Train")
y_train_valid = y_train[train_valid]

print("\n  Extracting TEST features...")
X_test, test_valid = extract_features_batch(paths_test, desc="  Test")
y_test_valid = y_test[test_valid]

print(f"\n  Feature vector size: {X_train.shape[1]}")
print(f"  Train features shape: {X_train.shape}")
print(f"  Test features shape:  {X_test.shape}")

# Feature scaling (StandardScaler)
print("\n  Scaling features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# SECTION 3: ML MODEL TRAINING & EVALUATION
# ============================================================================
print("\n[SECTION 3] Training ML Classifiers...")
print("-" * 50)

# Define classifiers as per the paper
classifiers = {
    'Random Forest (RF)': RandomForestClassifier(
        n_estimators=100,
        criterion='gini',          # Gini Index as per Eq.1 in paper
        class_weight='balanced',   # Handle class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    'Naive Bayes (NB)': GaussianNB(),  # Bayes theorem as per Eq.2 in paper
    'KNN': KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',        # Distance-weighted for imbalance handling
        n_jobs=-1
    ),
    'SVM': SVC(
        kernel='rbf',              # RBF kernel as per paper's mention
        class_weight='balanced',   # Handle class imbalance
        C=10,
        gamma='scale',
        random_state=RANDOM_STATE
    )
}

results = {}
predictions = {}
class_names = ['Authentic', 'Counterfeit']

for name, clf in classifiers.items():
    print(f"\n  Training {name}...")
    start_time = time.time()
    
    clf.fit(X_train_scaled, y_train_valid)
    y_pred = clf.predict(X_test_scaled)
    
    elapsed = time.time() - start_time
    acc = accuracy_score(y_test_valid, y_pred) * 100
    
    results[name] = {
        'accuracy': acc,
        'time': elapsed,
        'report': classification_report(y_test_valid, y_pred, 
                                         target_names=class_names),
        'confusion_matrix': confusion_matrix(y_test_valid, y_pred)
    }
    predictions[name] = y_pred
    
    print(f"    Accuracy: {acc:.2f}%")
    print(f"    Time: {elapsed:.2f}s")

# ============================================================================
# SECTION 4: RESULTS COMPARISON (Replicating Table 2 from paper)
# ============================================================================
print("\n[SECTION 4] Results Comparison")
print("=" * 70)

# Print Table 2 equivalent
print("\n  Table 2: Accuracy of the Counterfeit Medicines")
print("  " + "-" * 60)
print(f"  {'Algorithm':<25} {'Samples Tested':<20} {'Accuracy (%)':<15}")
print("  " + "-" * 60)

for name, res in results.items():
    n_samples = len(y_test_valid)
    print(f"  {name:<25} {n_samples:<20} {res['accuracy']:.2f}")

print("  " + "-" * 60)

# Print detailed classification reports
for name, res in results.items():
    print(f"\n  --- {name} Classification Report ---")
    print(res['report'])

# ============================================================================
# SECTION 5: VISUALIZATIONS
# ============================================================================
print("\n[SECTION 5] Generating Visualizations...")
print("-" * 50)

output_dir = os.path.join(BASE_DIR, "results")
os.makedirs(output_dir, exist_ok=True)

# --- Plot 1: Accuracy Comparison Bar Chart (replicating Graph 1 from paper) ---
fig, ax = plt.subplots(figsize=(10, 6))
algo_names = list(results.keys())
accuracies = [results[n]['accuracy'] for n in algo_names]
colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']

bars = ax.bar(algo_names, accuracies, color=colors, edgecolor='black', linewidth=0.8)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_xlabel('ML Algorithm', fontsize=13)
ax.set_title('Counterfeit Medicine Detection - Accuracy Comparison\n(Replicating Graph 1 from Paper)', 
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=150)
print(f"  Saved: {os.path.join(output_dir, 'accuracy_comparison.png')}")
plt.close()

# --- Plot 2: Confusion Matrices (2x2 grid) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, res) in enumerate(results.items()):
    cm = res['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[i], cbar=False)
    axes[i].set_title(f'{name}\nAccuracy: {res["accuracy"]:.2f}%', fontweight='bold')
    axes[i].set_ylabel('True Label')
    axes[i].set_xlabel('Predicted Label')

plt.suptitle('Confusion Matrices for All Classifiers', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=150)
print(f"  Saved: {os.path.join(output_dir, 'confusion_matrices.png')}")
plt.close()

# --- Plot 3: Class Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Train distribution
train_counts = [np.sum(y_train_valid == 0), np.sum(y_train_valid == 1)]
axes[0].bar(class_names, train_counts, color=['#27ae60', '#c0392b'], edgecolor='black')
for j, v in enumerate(train_counts):
    axes[0].text(j, v + 20, str(v), ha='center', fontweight='bold')
axes[0].set_title('Training Set Distribution', fontweight='bold')
axes[0].set_ylabel('Number of Samples')

# Test distribution
test_counts = [np.sum(y_test_valid == 0), np.sum(y_test_valid == 1)]
axes[1].bar(class_names, test_counts, color=['#27ae60', '#c0392b'], edgecolor='black')
for j, v in enumerate(test_counts):
    axes[1].text(j, v + 5, str(v), ha='center', fontweight='bold')
axes[1].set_title('Test Set Distribution', fontweight='bold')
axes[1].set_ylabel('Number of Samples')

plt.suptitle('Class Distribution After Stratified Split', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150)
print(f"  Saved: {os.path.join(output_dir, 'class_distribution.png')}")
plt.close()

# --- Plot 4: Feature Importance from Random Forest ---
rf_clf = classifiers['Random Forest (RF)']
importances = rf_clf.feature_importances_

# Create feature group labels
n_color = 512
n_lbp = 26
n_glcm = 5
n_hu = 7
n_hog = len(importances) - n_color - n_lbp - n_glcm - n_hu

group_names = ['Color Histogram', 'LBP Texture', 'GLCM Texture', 'Hu Moments', 'HOG']
group_sizes = [n_color, n_lbp, n_glcm, n_hu, n_hog]
group_importances = []

idx = 0
for size in group_sizes:
    group_importances.append(np.sum(importances[idx:idx+size]))
    idx += size

fig, ax = plt.subplots(figsize=(10, 6))
colors_feat = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c']
bars = ax.barh(group_names, group_importances, color=colors_feat, edgecolor='black')

for bar, imp in zip(bars, group_importances):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2.,
            f'{imp:.4f}', va='center', fontweight='bold')

ax.set_xlabel('Total Feature Importance', fontsize=12)
ax.set_title('Feature Group Importance (Random Forest)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
print(f"  Saved: {os.path.join(output_dir, 'feature_importance.png')}")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print(" FINAL SUMMARY")
print("=" * 70)

best_algo = max(results, key=lambda x: results[x]['accuracy'])
print(f"\n  Best Algorithm: {best_algo}")
print(f"  Best Accuracy:  {results[best_algo]['accuracy']:.2f}%")
print(f"\n  Ranking (highest to lowest accuracy):")

sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
for rank, (name, res) in enumerate(sorted_results, 1):
    print(f"    {rank}. {name}: {res['accuracy']:.2f}%")

print(f"\n  All results saved to: {output_dir}")
print("=" * 70)
print("  Implementation Complete!")
print("=" * 70)
