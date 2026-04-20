"""
=============================================================================
CONTRIBUTION: Dual-Stream Counterfeit Medicine Detection
=============================================================================

Base Paper: "Analyzing the Counterfeit Medicines Based on Classification
            Using Machine Learning Techniques"
            - Binitha S. Thomson and W. Rose Varuna (Springer, 2024)

Our 3 Contributions:
  1. DUAL-STREAM ARCHITECTURE: Visual features (Stream A) + OCR text
     features (Stream B) fused together for better detection
  2. XGBoost CLASSIFIER: Added as 5th classifier (outperforms SVM)
  3. SHAP EXPLAINABILITY: Interpretable analysis of feature importance

Pipeline:
  1. Load saved pipeline data (visual features + labels from base paper)
  2. Load pre-extracted OCR features (Stream B)
  3. Create 3 feature configurations: Visual-Only / OCR-Only / Fused
  4. Train 5 ML classifiers on all 3 configs (15 experiments)
  5. Compare results — show fused > visual-only
  6. SHAP explainability analysis on best model
  7. Independent test verification with dual-stream
  8. Generate all publication-ready charts and tables
=============================================================================
"""

import os
import sys
import warnings
import time
import pickle
import numpy as np
from collections import Counter
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, 'results')
RANDOM_STATE = 42
IMG_SIZE     = 128

CLASS_NAMES = ['Authentic', 'Counterfeit']

np.random.seed(RANDOM_STATE)
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
            print(f"  GPU: {name} ({mem:.1f} GB)")
            return 'cuda'
    except Exception:
        pass
    try:
        import xgboost as xgb
        t = xgb.XGBClassifier(device='cuda', n_estimators=1, verbosity=0)
        t.fit(np.array([[1,2],[3,4]], dtype=np.float32), np.array([0,1]))
        print("  GPU detected via XGBoost.")
        return 'cuda'
    except Exception:
        pass
    print("  No GPU. Using CPU.")
    return 'cpu'


# ============================================================================
# SECTION 1: LOAD SAVED PIPELINE DATA
# ============================================================================
def load_pipeline_data():
    """Load visual features and labels from the base paper pipeline."""
    print("\n" + "=" * 70)
    print("  SECTION 1: LOADING PIPELINE DATA")
    print("=" * 70)

    pkl_path = os.path.join(BASE_DIR, 'pipeline_data.pkl')
    if not os.path.exists(pkl_path):
        print(f"  ERROR: {pkl_path} not found!")
        print(f"  Run research_paper_implementation.py first.")
        sys.exit(1)

    data = pickle.load(open(pkl_path, 'rb'))
    print(f"  Loaded: {pkl_path}")
    print(f"  Keys: {list(data.keys())}")

    X_train_vis = data['X_train']        # (6657, 2314) visual features
    X_test_vis  = data['X_test']         # (1665, 2314)
    y_train     = data['y_tr']
    y_test      = data['y_te']
    paths_train = data['paths_train']
    paths_test  = data['paths_test']

    print(f"\n  Visual features:")
    print(f"    Train: {X_train_vis.shape}  "
          f"(auth={np.sum(y_train==0)}, fake={np.sum(y_train==1)})")
    print(f"    Test:  {X_test_vis.shape}  "
          f"(auth={np.sum(y_test==0)}, fake={np.sum(y_test==1)})")

    return X_train_vis, X_test_vis, y_train, y_test, paths_train, paths_test


# ============================================================================
# SECTION 2: LOAD OCR FEATURES (Stream B)
# ============================================================================
def load_ocr_features():
    """Load pre-extracted OCR features."""
    print("\n" + "=" * 70)
    print("  SECTION 2: LOADING OCR FEATURES (Stream B)")
    print("=" * 70)

    ocr_path = os.path.join(BASE_DIR, 'ocr_features.pkl')
    if not os.path.exists(ocr_path):
        print(f"  ERROR: {ocr_path} not found!")
        print(f"  Need to extract OCR features first.")
        sys.exit(1)

    ocr_data = pickle.load(open(ocr_path, 'rb'))
    X_train_ocr = ocr_data['X_train_ocr']   # (6657, 8)
    X_test_ocr  = ocr_data['X_test_ocr']    # (1665, 8)

    print(f"  Loaded: {ocr_path}")
    print(f"  OCR features:")
    print(f"    Train: {X_train_ocr.shape}")
    print(f"    Test:  {X_test_ocr.shape}")

    ocr_feature_names = [
        'n_text_regions', 'total_chars', 'avg_confidence',
        'max_confidence', 'min_confidence', 'confidence_std',
        'text_area_ratio', 'char_density'
    ]
    print(f"  Features: {ocr_feature_names}")

    return X_train_ocr, X_test_ocr, ocr_feature_names


# ============================================================================
# SECTION 3: CREATE 3 FEATURE CONFIGURATIONS
# ============================================================================
def create_feature_configs(X_train_vis, X_test_vis, X_train_ocr, X_test_ocr):
    """
    Create 3 experiment configurations:
      A: Visual Only  (2314 features)
      B: OCR Only     (8 features)
      C: Fused        (2322 features)
    Each gets its own StandardScaler.
    """
    print("\n" + "=" * 70)
    print("  SECTION 3: FEATURE FUSION — 3 CONFIGURATIONS")
    print("=" * 70)

    configs = {}

    # Config A: Visual Only
    scaler_a = StandardScaler()
    X_tr_a = scaler_a.fit_transform(X_train_vis).astype(np.float32)
    X_te_a = scaler_a.transform(X_test_vis).astype(np.float32)
    configs['Visual Only'] = {
        'X_train': X_tr_a, 'X_test': X_te_a,
        'scaler': scaler_a, 'n_features': X_tr_a.shape[1]
    }
    print(f"  Config A — Visual Only: {X_tr_a.shape[1]} features")

    # Config B: OCR Only
    scaler_b = StandardScaler()
    X_tr_b = scaler_b.fit_transform(X_train_ocr).astype(np.float32)
    X_te_b = scaler_b.transform(X_test_ocr).astype(np.float32)
    configs['OCR Only'] = {
        'X_train': X_tr_b, 'X_test': X_te_b,
        'scaler': scaler_b, 'n_features': X_tr_b.shape[1]
    }
    print(f"  Config B — OCR Only:    {X_tr_b.shape[1]} features")

    # Config C: Fused (Visual + OCR)
    X_train_fused = np.hstack([X_train_vis, X_train_ocr])
    X_test_fused  = np.hstack([X_test_vis, X_test_ocr])
    scaler_c = StandardScaler()
    X_tr_c = scaler_c.fit_transform(X_train_fused).astype(np.float32)
    X_te_c = scaler_c.transform(X_test_fused).astype(np.float32)
    configs['Fused'] = {
        'X_train': X_tr_c, 'X_test': X_te_c,
        'scaler': scaler_c, 'n_features': X_tr_c.shape[1]
    }
    print(f"  Config C — Fused:       {X_tr_c.shape[1]} features")

    return configs


# ============================================================================
# SECTION 4: BUILD CLASSIFIERS
# ============================================================================
def build_classifiers(y_train, gpu_device):
    """Build all 5 classifiers."""
    spw = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)

    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, criterion='gini', class_weight='balanced',
            max_depth=20, min_samples_split=10, min_samples_leaf=4,
            random_state=RANDOM_STATE, n_jobs=-1),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(
            n_neighbors=5, weights='distance', n_jobs=-1),
        'SVM': SVC(
            kernel='rbf', class_weight='balanced', C=1.0, gamma='scale',
            random_state=RANDOM_STATE),
        'XGBoost': XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=2.0, gamma=0.1,
            scale_pos_weight=spw, eval_metric='logloss',
            tree_method='hist', device=gpu_device,
            random_state=RANDOM_STATE, n_jobs=-1),
    }


# ============================================================================
# SECTION 5: TRAIN ALL CONFIGURATIONS (5 models × 3 configs = 15 runs)
# ============================================================================
def train_all_configs(configs, y_train, y_test, gpu_device):
    """Train 5 classifiers on all 3 feature configs."""
    print("\n" + "=" * 70)
    print("  SECTION 4: TRAINING — 5 models × 3 configs = 15 experiments")
    print("=" * 70)

    all_results = {}     # {config_name: {model_name: metrics_dict}}
    all_models  = {}     # {config_name: {model_name: trained_model}}

    for config_name, cfg in configs.items():
        print(f"\n  {'─' * 60}")
        print(f"  Config: {config_name} ({cfg['n_features']} features)")
        print(f"  {'─' * 60}")

        classifiers = build_classifiers(y_train, gpu_device)
        config_results = {}
        config_models  = {}

        for model_name, clf in classifiers.items():
            print(f"\n    Training {model_name}...")
            t0 = time.time()
            clf.fit(cfg['X_train'], y_train)
            y_pred = clf.predict(cfg['X_test'])
            elapsed = time.time() - t0

            acc  = accuracy_score(y_test, y_pred) * 100
            prec = precision_score(y_test, y_pred, average='weighted') * 100
            rec  = recall_score(y_test, y_pred, average='weighted') * 100
            f1   = f1_score(y_test, y_pred, average='weighted') * 100

            if np.sum(y_test == 1) > 0:
                fake_rec  = recall_score(y_test, y_pred, pos_label=1) * 100
                fake_prec = precision_score(y_test, y_pred, pos_label=1) * 100
                fake_f1   = f1_score(y_test, y_pred, pos_label=1) * 100
            else:
                fake_rec = fake_prec = fake_f1 = 0.0

            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(
                y_test, y_pred, target_names=CLASS_NAMES, digits=4)

            config_results[model_name] = {
                'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                'fake_recall': fake_rec, 'fake_precision': fake_prec,
                'fake_f1': fake_f1, 'time': elapsed,
                'confusion_matrix': cm, 'report': report, 'y_pred': y_pred,
            }
            config_models[model_name] = clf

            print(f"      Acc: {acc:.2f}%  |  Fake Recall: {fake_rec:.2f}%  "
                  f"|  Fake F1: {fake_f1:.2f}%  |  {elapsed:.2f}s")

        all_results[config_name] = config_results
        all_models[config_name] = config_models

    return all_results, all_models


# ============================================================================
# SECTION 6: RESULTS COMPARISON TABLE
# ============================================================================
def print_comparison_table(all_results):
    """Print the master comparison table: 5 models × 3 configs."""
    print("\n" + "=" * 70)
    print("  SECTION 5: RESULTS COMPARISON TABLE")
    print("=" * 70)

    config_names = list(all_results.keys())

    # Table header
    header = f"\n  {'Algorithm':<18}"
    for cfg in config_names:
        header += f"  {cfg + ' Acc%':<16}"
    header += f"  {'Δ (Fused-Visual)':<18}"
    print(header)
    print("  " + "─" * 85)

    for model_name in all_results[config_names[0]].keys():
        row = f"  {model_name:<18}"
        accs = {}
        for cfg in config_names:
            acc = all_results[cfg][model_name]['accuracy']
            accs[cfg] = acc
            row += f"  {acc:<16.2f}"

        # Delta: Fused - Visual Only
        if 'Fused' in accs and 'Visual Only' in accs:
            delta = accs['Fused'] - accs['Visual Only']
            sign = '+' if delta >= 0 else ''
            row += f"  {sign}{delta:<17.2f}"

        print(row)

    print("  " + "─" * 85)

    # Counterfeit recall table
    print(f"\n  {'Algorithm':<18}", end='')
    for cfg in config_names:
        print(f"  {cfg + ' FakeRec%':<16}", end='')
    print(f"  {'Δ (Fused-Visual)':<18}")
    print("  " + "─" * 85)

    for model_name in all_results[config_names[0]].keys():
        row = f"  {model_name:<18}"
        recs = {}
        for cfg in config_names:
            rec = all_results[cfg][model_name]['fake_recall']
            recs[cfg] = rec
            row += f"  {rec:<16.2f}"

        if 'Fused' in recs and 'Visual Only' in recs:
            delta = recs['Fused'] - recs['Visual Only']
            sign = '+' if delta >= 0 else ''
            row += f"  {sign}{delta:<17.2f}"
        print(row)

    print("  " + "─" * 85)

    # Print detailed reports for Fused config
    print("\n  Detailed Classification Reports (Fused Config):")
    for model_name, res in all_results.get('Fused', {}).items():
        print(f"\n  --- {model_name} ---")
        print(res['report'])


# ============================================================================
# SECTION 7: VISUALIZATIONS
# ============================================================================
def generate_visualizations(all_results, y_test, output_dir):
    """Generate all comparison charts."""
    print("\n" + "=" * 70)
    print("  SECTION 6: GENERATING VISUALIZATIONS")
    print("=" * 70)

    config_names = list(all_results.keys())
    model_names  = list(all_results[config_names[0]].keys())
    colors_config = {'Visual Only': '#3498db', 'OCR Only': '#2ecc71', 'Fused': '#e74c3c'}

    # ---- Plot 1: Grouped Bar Chart — Accuracy per model × config ----
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(model_names))
    width = 0.25

    for i, cfg in enumerate(config_names):
        accs = [all_results[cfg][m]['accuracy'] for m in model_names]
        bars = ax.bar(x + i * width, accs, width, label=cfg,
                      color=colors_config[cfg], edgecolor='black', linewidth=0.5)
        for bar, a in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, a + 0.3,
                    f'{a:.1f}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_xlabel('ML Algorithm', fontsize=13)
    ax.set_title('Dual-Stream Counterfeit Detection — Accuracy Comparison\n'
                 'Visual Only vs OCR Only vs Fused',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dual_stream_accuracy.png'), dpi=150)
    plt.close()
    print("  Saved: dual_stream_accuracy.png")

    # ---- Plot 2: Heatmap — 5 models × 3 configs ----
    acc_matrix = np.array([
        [all_results[cfg][m]['accuracy'] for cfg in config_names]
        for m in model_names
    ])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(acc_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=config_names, yticklabels=model_names,
                ax=ax, cbar_kws={'label': 'Accuracy (%)'},
                annot_kws={'size': 14, 'fontweight': 'bold'},
                linewidths=1, linecolor='white')
    ax.set_title('Accuracy Heatmap — 5 Models × 3 Configurations',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'config_heatmap.png'), dpi=150)
    plt.close()
    print("  Saved: config_heatmap.png")

    # ---- Plot 3: Improvement Chart — Fused vs Visual Only ----
    fig, ax = plt.subplots(figsize=(12, 6))
    deltas = []
    for m in model_names:
        fused_acc  = all_results['Fused'][m]['accuracy']
        visual_acc = all_results['Visual Only'][m]['accuracy']
        deltas.append(fused_acc - visual_acc)

    bar_colors = ['#27ae60' if d >= 0 else '#e74c3c' for d in deltas]
    bars = ax.bar(model_names, deltas, color=bar_colors, edgecolor='black')
    for bar, d in zip(bars, deltas):
        sign = '+' if d >= 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.1 if d >= 0 else -0.3),
                f'{sign}{d:.2f}%', ha='center', fontweight='bold', fontsize=12)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Accuracy Improvement (%)', fontsize=13)
    ax.set_xlabel('ML Algorithm', fontsize=13)
    ax.set_title('Improvement: Fused (Visual + OCR) vs Visual Only\n'
                 'Green = OCR features helped',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_chart.png'), dpi=150)
    plt.close()
    print("  Saved: improvement_chart.png")

    # ---- Plot 4: Counterfeit Recall Comparison ----
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, cfg in enumerate(config_names):
        recs = [all_results[cfg][m]['fake_recall'] for m in model_names]
        bars = ax.bar(x + i * width, recs, width, label=cfg,
                      color=colors_config[cfg], edgecolor='black', linewidth=0.5)
        for bar, r in zip(bars, recs):
            ax.text(bar.get_x() + bar.get_width()/2, r + 0.3,
                    f'{r:.1f}', ha='center', fontsize=8, fontweight='bold')

    ax.set_ylabel('Counterfeit Recall (%)', fontsize=13)
    ax.set_xlabel('ML Algorithm', fontsize=13)
    ax.set_title('Counterfeit Detection Recall — All Configurations',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'counterfeit_recall_comparison.png'), dpi=150)
    plt.close()
    print("  Saved: counterfeit_recall_comparison.png")

    # ---- Plot 5: Confusion Matrices for Fused config ----
    fused = all_results['Fused']
    n_clf = len(fused)
    n_cols = 3
    n_rows = (n_clf + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (name, r) in enumerate(fused.items()):
        row, col = divmod(i, n_cols)
        cm = r['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=axes[row][col], cbar=False, annot_kws={'size': 14})
        axes[row][col].set_title(
            f"{name}\nAcc: {r['accuracy']:.2f}% | "
            f"Fake Recall: {r['fake_recall']:.1f}%",
            fontweight='bold', fontsize=10)
        axes[row][col].set_ylabel('True')
        axes[row][col].set_xlabel('Predicted')

    for j in range(n_clf, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row][col].set_visible(False)

    plt.suptitle('Confusion Matrices — Fused Configuration (Visual + OCR)',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_fused.png'), dpi=150)
    plt.close()
    print("  Saved: confusion_matrices_fused.png")

    # ---- Plot 6: Radar Chart — Best model multi-metric comparison ----
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'Fake Recall', 'Fake F1']
    best_model = max(all_results['Fused'],
                     key=lambda m: all_results['Fused'][m]['accuracy'])
    print(f"\n  Best Fused model: {best_model}")

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]

    for cfg_name in config_names:
        r = all_results[cfg_name][best_model]
        values = [r['accuracy'], r['precision'], r['recall'],
                  r['f1'], r['fake_recall'], r['fake_f1']]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2,
                label=cfg_name, color=colors_config[cfg_name])
        ax.fill(angles, values, alpha=0.1, color=colors_config[cfg_name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title(f'Multi-Metric Radar — {best_model}\n'
                 f'Visual vs OCR vs Fused',
                 fontweight='bold', fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_chart.png'), dpi=150)
    plt.close()
    print("  Saved: radar_chart.png")


# ============================================================================
# SECTION 8: SHAP EXPLAINABILITY
# ============================================================================
def run_shap_analysis(all_models, configs, y_test, ocr_feature_names, output_dir):
    """Run SHAP explainability analysis on the best fused model."""
    print("\n" + "=" * 70)
    print("  SECTION 7: SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 70)

    try:
        import shap
    except ImportError:
        print("  SHAP not installed. Run: pip install shap")
        print("  Skipping SHAP analysis.")
        return

    # Use XGBoost (fused) for SHAP — tree-based = fast exact SHAP
    fused_models = all_models.get('Fused', {})
    xgb_model = fused_models.get('XGBoost')
    rf_model  = fused_models.get('Random Forest')

    if xgb_model is None:
        print("  No XGBoost model found. Skipping.")
        return

    X_test_fused = configs['Fused']['X_test']

    # Build feature names
    n_color, n_lbp, n_glcm, n_hu = 512, 26, 5, 7
    n_hog = X_test_fused.shape[1] - n_color - n_lbp - n_glcm - n_hu - 8
    feature_names = (
        [f'color_{i}' for i in range(n_color)] +
        [f'lbp_{i}' for i in range(n_lbp)] +
        [f'glcm_{i}' for i in range(n_glcm)] +
        [f'hu_{i}' for i in range(n_hu)] +
        [f'hog_{i}' for i in range(n_hog)] +
        ocr_feature_names
    )

    # Feature group mapping
    group_map = {}
    idx = 0
    for gname, gsize in [('Color Histogram', n_color), ('LBP Texture', n_lbp),
                          ('GLCM Texture', n_glcm), ('Hu Moments', n_hu),
                          ('HOG', n_hog), ('OCR', 8)]:
        for j in range(gsize):
            group_map[idx + j] = gname
        idx += gsize

    # ---- SHAP TreeExplainer ----
    print("\n  Computing SHAP values for XGBoost (Fused)...")
    t0 = time.time()
    explainer = shap.TreeExplainer(xgb_model)

    # Use a subset for speed (500 samples)
    n_shap = min(500, len(X_test_fused))
    X_shap = X_test_fused[:n_shap]
    shap_values = explainer.shap_values(X_shap)
    print(f"  SHAP computed in {time.time()-t0:.1f}s for {n_shap} samples")

    # ---- SHAP Plot 1: Global Summary (Top 20 features) ----
    print("  Generating SHAP summary plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_shap,
                      feature_names=feature_names,
                      max_display=20, show=False)
    plt.title('SHAP Feature Importance — Top 20 Features (XGBoost Fused)',
              fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved: shap_summary.png")

    # ---- SHAP Plot 2: Feature Group Importance ----
    print("  Computing feature group importance...")
    abs_shap = np.abs(shap_values)
    mean_abs_shap = abs_shap.mean(axis=0)

    group_importance = {}
    for feat_idx, group in group_map.items():
        if feat_idx < len(mean_abs_shap):
            group_importance[group] = (
                group_importance.get(group, 0) + mean_abs_shap[feat_idx])

    groups = list(group_importance.keys())
    importances = list(group_importance.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_groups = {
        'Color Histogram': '#3498db', 'LBP Texture': '#e67e22',
        'GLCM Texture': '#2ecc71', 'Hu Moments': '#9b59b6',
        'HOG': '#e74c3c', 'OCR': '#f39c12'
    }
    bar_colors = [colors_groups.get(g, '#95a5a6') for g in groups]
    bars = ax.barh(groups, importances, color=bar_colors, edgecolor='black')
    for bar, v in zip(bars, importances):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', fontweight='bold')
    ax.set_xlabel('Mean |SHAP value|', fontsize=12)
    ax.set_title('Feature Group Importance (SHAP — XGBoost Fused)',
                 fontweight='bold', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_feature_groups.png'), dpi=150)
    plt.close()
    print("  Saved: shap_feature_groups.png")

    # ---- SHAP Plot 3: Stream A vs Stream B Pie ----
    visual_importance = sum(v for k, v in group_importance.items() if k != 'OCR')
    ocr_importance = group_importance.get('OCR', 0)
    total = visual_importance + ocr_importance

    fig, ax = plt.subplots(figsize=(8, 8))
    sizes = [visual_importance, ocr_importance]
    labels_pie = [
        f'Stream A: Visual\n{visual_importance/total*100:.1f}%',
        f'Stream B: OCR\n{ocr_importance/total*100:.1f}%'
    ]
    pie_colors = ['#3498db', '#f39c12']
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels_pie, colors=pie_colors,
        autopct='', startangle=90, textprops={'fontsize': 14},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    ax.set_title('Stream Contribution to Classification\n'
                 'Visual Features vs OCR Features',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_stream_comparison.png'), dpi=150)
    plt.close()
    print("  Saved: shap_stream_comparison.png")

    # ---- SHAP Plot 4: OCR Feature Dependence (avg_confidence) ----
    ocr_start_idx = n_color + n_lbp + n_glcm + n_hu + n_hog
    avg_conf_idx = ocr_start_idx + 2  # avg_confidence is 3rd OCR feature

    if avg_conf_idx < X_shap.shape[1]:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            avg_conf_idx, shap_values, X_shap,
            feature_names=feature_names,
            show=False, ax=ax)
        ax.set_title('SHAP Dependence: OCR avg_confidence\n'
                     'Does low text confidence → counterfeit?',
                     fontweight='bold', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_dependence.png'), dpi=150)
        plt.close()
        print("  Saved: shap_dependence.png")


# ============================================================================
# SECTION 9: INDEPENDENT TEST VERIFICATION
# ============================================================================
def verify_independent_test(all_models, configs, ocr_feature_names, output_dir):
    """Test Fused model on independent unseen images."""
    print("\n" + "=" * 70)
    print("  SECTION 8: INDEPENDENT TEST VERIFICATION")
    print("=" * 70)

    import cv2
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog

    ind_test_dir = os.path.join(BASE_DIR, 'ind_test')
    if not os.path.isdir(ind_test_dir):
        print("  No ind_test/ directory found. Skipping.")
        return

    exts = ('.jpg', '.jpeg', '.png')
    test_images = [f for f in os.listdir(ind_test_dir) if f.lower().endswith(exts)]
    if not test_images:
        print("  No test images found.")
        return

    print(f"  Found {len(test_images)} independent test images")

    # Feature extraction functions (same as base paper)
    def extract_visual_features(img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color = cv2.normalize(
            cv2.calcHist([hsv], [0,1,2], None, [8,8,8],
                         [0,180,0,256,0,256]),
            None).flatten()

        r, n = 3, 24
        lbp_img = local_binary_pattern(gray, n, r, method='uniform')
        lbp_h, _ = np.histogram(lbp_img.ravel(), bins=n+2,
                                range=(0, n+2), density=True)

        gq = (gray // 4).astype(np.uint8)
        glcm = graycomatrix(gq, [1,3], [0, np.pi/4, np.pi/2],
                            64, symmetric=True, normed=True)
        glcm_f = np.array([graycoprops(glcm, p).mean()
                           for p in ['contrast','dissimilarity',
                                     'homogeneity','energy','correlation']])

        hu = cv2.HuMoments(cv2.moments(gray)).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        hog_f = hog(gray, orientations=9, pixels_per_cell=(16,16),
                    cells_per_block=(2,2), feature_vector=True)

        return np.concatenate([color, lbp_h, glcm_f, hu, hog_f])

    def extract_ocr_features(img_path):
        """Placeholder OCR features (zeros if easyocr not available)."""
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            img = cv2.imread(img_path)
            if img is None:
                return np.zeros(8)
            results = reader.readtext(img)

            if not results:
                return np.zeros(8)

            confidences = [r[2] for r in results]
            texts = [r[1] for r in results]
            total_chars = sum(len(t) for t in texts)
            h, w = img.shape[:2]
            img_area = h * w

            bbox_areas = []
            for r in results:
                pts = np.array(r[0])
                bw = np.linalg.norm(pts[1] - pts[0])
                bh = np.linalg.norm(pts[3] - pts[0])
                bbox_areas.append(bw * bh)
            sum_area = sum(bbox_areas)

            return np.array([
                len(results),                           # n_text_regions
                total_chars,                            # total_chars
                np.mean(confidences),                   # avg_confidence
                np.max(confidences),                    # max_confidence
                np.min(confidences),                    # min_confidence
                np.std(confidences) if len(confidences) > 1 else 0,
                sum_area / max(img_area, 1),            # text_area_ratio
                total_chars / max(sum_area, 1),         # char_density
            ])
        except ImportError:
            return np.zeros(8)

    # Get the fused scaler
    fused_scaler = configs['Fused']['scaler']

    # Compare Visual-only vs Fused predictions
    print(f"\n  {'Image':<35} {'Visual-Only':<15} {'Fused':<15}")
    print("  " + "─" * 65)

    visual_model = all_models['Visual Only'].get('XGBoost')
    fused_model  = all_models['Fused'].get('XGBoost')
    visual_scaler = configs['Visual Only']['scaler']

    if visual_model is None or fused_model is None:
        print("  Models not available.")
        return

    for img_file in test_images[:10]:
        img_path = os.path.join(ind_test_dir, img_file)
        vis_feat = extract_visual_features(img_path)
        ocr_feat = extract_ocr_features(img_path)

        if vis_feat is None:
            continue

        # Visual-only prediction
        vis_scaled = visual_scaler.transform(vis_feat.reshape(1, -1))
        vis_pred = CLASS_NAMES[visual_model.predict(vis_scaled)[0]]

        # Fused prediction
        fused_feat = np.concatenate([vis_feat, ocr_feat])
        fused_scaled = fused_scaler.transform(fused_feat.reshape(1, -1))
        fused_pred = CLASS_NAMES[fused_model.predict(fused_scaled)[0]]

        print(f"  {img_file:<35} {vis_pred:<15} {fused_pred:<15}")


# ============================================================================
# SECTION 10: FINAL SUMMARY
# ============================================================================
def print_final_summary(all_results):
    """Print contribution summary."""
    print("\n" + "=" * 70)
    print("  SECTION 9: CONTRIBUTION SUMMARY")
    print("=" * 70)

    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Aspect           │  Base Paper          │  Our Contribution │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print("  │  Features         │  Visual only         │  Visual + OCR     │")
    print("  │  Models           │  RF, NB, KNN, SVM    │  + XGBoost        │")
    print("  │  Explainability   │  None                │  SHAP analysis    │")
    print("  │  Class Balance    │  None                │  Augmentation     │")
    print("  └─────────────────────────────────────────────────────────────┘")

    # Find biggest improvements
    fused = all_results.get('Fused', {})
    visual = all_results.get('Visual Only', {})

    if fused and visual:
        print("\n  Key Findings:")

        best_fused = max(fused, key=lambda m: fused[m]['accuracy'])
        best_visual = max(visual, key=lambda m: visual[m]['accuracy'])

        print(f"\n    Best Fused Model:       {best_fused}")
        print(f"      Accuracy:             {fused[best_fused]['accuracy']:.2f}%")
        print(f"      Counterfeit Recall:   {fused[best_fused]['fake_recall']:.2f}%")
        print(f"      Counterfeit F1:       {fused[best_fused]['fake_f1']:.2f}%")

        print(f"\n    Best Visual-Only Model: {best_visual}")
        print(f"      Accuracy:             {visual[best_visual]['accuracy']:.2f}%")

        for model_name in fused:
            if model_name in visual:
                delta = fused[model_name]['accuracy'] - visual[model_name]['accuracy']
                sign = '+' if delta >= 0 else ''
                print(f"\n    {model_name}: Visual={visual[model_name]['accuracy']:.2f}% "
                      f"→ Fused={fused[model_name]['accuracy']:.2f}% "
                      f"({sign}{delta:.2f}%)")

    print("\n  " + "=" * 55)
    print("  Contribution narrative:")
    print("  \"We extend the base paper's approach in three ways:")
    print("    (1) A dual-stream architecture combining visual features")
    print("        with OCR-based text analysis,")
    print("    (2) XGBoost as a 5th classifier,")
    print("    (3) SHAP explainability for interpretable insights.\"")
    print("  " + "=" * 55)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("  DUAL-STREAM COUNTERFEIT MEDICINE DETECTION")
    print("  Our Contribution: Visual + OCR Fusion + XGBoost + SHAP")
    print("=" * 70)

    gpu_device = detect_gpu()

    # Load data
    X_train_vis, X_test_vis, y_train, y_test, paths_train, paths_test = \
        load_pipeline_data()
    X_train_ocr, X_test_ocr, ocr_feature_names = load_ocr_features()

    # Verify alignment
    assert X_train_vis.shape[0] == X_train_ocr.shape[0], \
        f"Train mismatch: {X_train_vis.shape[0]} vs {X_train_ocr.shape[0]}"
    assert X_test_vis.shape[0] == X_test_ocr.shape[0], \
        f"Test mismatch: {X_test_vis.shape[0]} vs {X_test_ocr.shape[0]}"

    # Create 3 configs
    configs = create_feature_configs(
        X_train_vis, X_test_vis, X_train_ocr, X_test_ocr)

    # Train all
    all_results, all_models = train_all_configs(
        configs, y_train, y_test, gpu_device)

    # Results
    print_comparison_table(all_results)

    # Visualizations
    generate_visualizations(all_results, y_test, OUTPUT_DIR)

    # SHAP
    run_shap_analysis(all_models, configs, y_test, ocr_feature_names, OUTPUT_DIR)

    # Independent test
    verify_independent_test(all_models, configs, ocr_feature_names, OUTPUT_DIR)

    # Summary
    print_final_summary(all_results)

    # Save everything
    print("\n" + "=" * 70)
    print("  SAVING CONTRIBUTION DATA")
    print("=" * 70)

    save_data = {
        'all_results': all_results,
        'all_models': all_models,
        'configs': {k: {'n_features': v['n_features']} for k, v in configs.items()},
        'y_train': y_train,
        'y_test': y_test,
        'class_names': CLASS_NAMES,
    }
    pkl_path = os.path.join(BASE_DIR, 'contribution_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved: {pkl_path} ({os.path.getsize(pkl_path)/1e6:.1f} MB)")

    print(f"\n  All charts saved to: {OUTPUT_DIR}")
    print("=" * 70)
    print("  Contribution Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
