"""
Generate 4.ipynb — Faithful reproduction of the WORKING original code
with RF, SVM, XGBoost comparison and ind_test evaluation.

KEY: Uses YOLO-predicted crops from runs/detect/predict/crops/ for training.
NO resize except gray->256x256 for GLCM. Threshold 0.30 for XGBoost.
"""

import json

def md(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source if isinstance(source, list) else [source]}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [],
            "source": source if isinstance(source, list) else [source]}

cells = []

# ── Title ──
cells.append(md([
    "# Counterfeit Medicine Detection Using Machine Learning\n",
    "\n",
    "**Reference Paper:** *Analyzing the Counterfeit Medicines Based on Classification Using Machine Learning Techniques*\n",
    "\n",
    "## Pipeline\n",
    "1. **YOLOv8** — Detect & crop medicine regions\n",
    "2. **Feature Extraction** — Color (6) + Texture/GLCM (4) + Shape (3) = 13 features\n",
    "3. **ML Classification** — Random Forest, SVM, XGBoost\n",
    "4. **Comparison** — Accuracy, Precision, Recall, F1-Score\n",
    "5. **Independent Testing** — `ind_test/` images"
]))

# ── Imports ──
cells.append(md(["## 1. Setup & Imports"]))
cells.append(code([
    "import os\n",
    "import cv2\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "from collections import Counter\n",
    "from ultralytics import YOLO\n",
    "from skimage.feature import graycomatrix, graycoprops\n",
    "\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.svm import SVC\n",
    "from xgboost import XGBClassifier\n",
    "\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score, precision_score, recall_score, f1_score,\n",
    "    classification_report, confusion_matrix, ConfusionMatrixDisplay\n",
    ")\n",
    "import joblib\n",
    "\n",
    "print('All imports successful.')"
]))

# ── Dataset Overview ──
cells.append(md(["## 2. Dataset Overview"]))
cells.append(code([
    "base_path = '.'\n",
    "splits = ['train', 'valid', 'test']\n",
    "\n",
    "for split in splits:\n",
    "    img_path = os.path.join(base_path, split, 'images')\n",
    "    lbl_path = os.path.join(base_path, split, 'labels')\n",
    "    num_images = len(os.listdir(img_path))\n",
    "    num_labels = len(os.listdir(lbl_path))\n",
    "    print(f'{split.upper()}: Images={num_images}, Labels={num_labels}')\n",
    "\n",
    "print()\n",
    "label_dir = 'train/labels'\n",
    "counter = Counter()\n",
    "for file in os.listdir(label_dir):\n",
    "    with open(os.path.join(label_dir, file), 'r') as f:\n",
    "        for line in f.readlines():\n",
    "            class_id = line.split()[0]\n",
    "            counter[class_id] += 1\n",
    "print('Class distribution in TRAIN:', counter)"
]))

# ── YOLO ──
cells.append(md([
    "## 3. YOLOv8 — Load Model & Generate Crops\n",
    "\n",
    "We use the pre-trained YOLOv8 model to detect medicine regions and save crops.\n",
    "The crops are saved to `runs/detect/predict/crops/authentic/` and `counterfeit/`."
]))
cells.append(code([
    "yolo_model = YOLO('runs/detect/train13/weights/best.pt')\n",
    "print('YOLO model loaded.')\n",
    "\n",
    "# Check if crops already exist\n",
    "crops_path = 'runs/detect/predict/crops'\n",
    "if os.path.exists(crops_path):\n",
    "    for cls_name in os.listdir(crops_path):\n",
    "        cls_path = os.path.join(crops_path, cls_name)\n",
    "        if os.path.isdir(cls_path):\n",
    "            print(f'  {cls_name}: {len(os.listdir(cls_path))} crops')\n",
    "    print('Using existing crops.')\n",
    "else:\n",
    "    print('Generating crops... (this may take a while)')\n",
    "    yolo_model.predict(\n",
    "        source='train/images',\n",
    "        save_crop=True,\n",
    "        save=True,\n",
    "        conf=0.25\n",
    "    )\n",
    "    print('Crops generated.')"
]))

# ── Color Features ──
cells.append(md([
    "## 4. Feature Extraction from YOLO Crops\n",
    "\n",
    "Extract **13 features** from each YOLO-detected crop:\n",
    "- **Color:** Mean & Std of R, G, B channels (6 features)\n",
    "- **Texture:** GLCM contrast, energy, homogeneity, correlation (4 features)\n",
    "- **Shape:** Contour area, perimeter, aspect ratio (3 features)\n",
    "\n",
    "### 4.1 Color Features"
]))
cells.append(code([
    "base_path = 'runs/detect/predict/crops'\n",
    "\n",
    "data = []\n",
    "\n",
    "for class_name in os.listdir(base_path):\n",
    "    class_path = os.path.join(base_path, class_name)\n",
    "    if not os.path.isdir(class_path):\n",
    "        continue\n",
    "\n",
    "    for img_file in os.listdir(class_path):\n",
    "        img_path = os.path.join(class_path, img_file)\n",
    "\n",
    "        img = cv2.imread(img_path)\n",
    "        if img is None:\n",
    "            continue\n",
    "        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "\n",
    "        # Mean and Std for each channel\n",
    "        mean = np.mean(img, axis=(0,1))\n",
    "        std = np.std(img, axis=(0,1))\n",
    "\n",
    "        features = [\n",
    "            mean[0], mean[1], mean[2],   # R, G, B mean\n",
    "            std[0], std[1], std[2],      # R, G, B std\n",
    "        ]\n",
    "\n",
    "        label = 0 if class_name == 'authentic' else 1\n",
    "        data.append(features + [label])\n",
    "\n",
    "columns = ['mean_R', 'mean_G', 'mean_B', 'std_R', 'std_G', 'std_B', 'label']\n",
    "df = pd.DataFrame(data, columns=columns)\n",
    "\n",
    "print(f'Color features extracted: {len(df)} samples')\n",
    "df.head()"
]))

# ── Texture Features ──
cells.append(md(["### 4.2 Texture Features (GLCM)"]))
cells.append(code([
    "texture_data = []\n",
    "\n",
    "for class_name in os.listdir(base_path):\n",
    "    class_path = os.path.join(base_path, class_name)\n",
    "    if not os.path.isdir(class_path):\n",
    "        continue\n",
    "\n",
    "    for img_file in os.listdir(class_path):\n",
    "        img_path = os.path.join(class_path, img_file)\n",
    "\n",
    "        img = cv2.imread(img_path)\n",
    "        if img is None:\n",
    "            continue\n",
    "        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n",
    "\n",
    "        # Resize ONLY for GLCM stability\n",
    "        gray = cv2.resize(gray, (256, 256))\n",
    "\n",
    "        glcm = graycomatrix(gray, distances=[1], angles=[0],\n",
    "                            levels=256, symmetric=True, normed=True)\n",
    "\n",
    "        contrast    = graycoprops(glcm, 'contrast')[0, 0]\n",
    "        energy      = graycoprops(glcm, 'energy')[0, 0]\n",
    "        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]\n",
    "        correlation = graycoprops(glcm, 'correlation')[0, 0]\n",
    "\n",
    "        label = 0 if class_name == 'authentic' else 1\n",
    "        texture_data.append([contrast, energy, homogeneity, correlation, label])\n",
    "\n",
    "texture_columns = ['contrast', 'energy', 'homogeneity', 'correlation', 'label']\n",
    "texture_df = pd.DataFrame(texture_data, columns=texture_columns)\n",
    "\n",
    "print(f'Texture features extracted: {len(texture_df)} samples')\n",
    "texture_df.head()"
]))

# ── Shape Features ──
cells.append(md(["### 4.3 Shape Features"]))
cells.append(code([
    "shape_data = []\n",
    "\n",
    "for class_name in os.listdir(base_path):\n",
    "    class_path = os.path.join(base_path, class_name)\n",
    "    if not os.path.isdir(class_path):\n",
    "        continue\n",
    "\n",
    "    for img_file in os.listdir(class_path):\n",
    "        img_path = os.path.join(class_path, img_file)\n",
    "\n",
    "        img = cv2.imread(img_path)\n",
    "        if img is None:\n",
    "            continue\n",
    "        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n",
    "\n",
    "        # Threshold to get object mask\n",
    "        _, thresh = cv2.threshold(gray, 127, 255,\n",
    "                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)\n",
    "\n",
    "        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,\n",
    "                                       cv2.CHAIN_APPROX_SIMPLE)\n",
    "\n",
    "        if len(contours) > 0:\n",
    "            cnt = max(contours, key=cv2.contourArea)\n",
    "            area = cv2.contourArea(cnt)\n",
    "            perimeter = cv2.arcLength(cnt, True)\n",
    "            x, y, w, h = cv2.boundingRect(cnt)\n",
    "            aspect_ratio = float(w) / h if h != 0 else 0\n",
    "        else:\n",
    "            area = 0\n",
    "            perimeter = 0\n",
    "            aspect_ratio = 0\n",
    "\n",
    "        label = 0 if class_name == 'authentic' else 1\n",
    "        shape_data.append([area, perimeter, aspect_ratio, label])\n",
    "\n",
    "shape_columns = ['area', 'perimeter', 'aspect_ratio', 'label']\n",
    "shape_df = pd.DataFrame(shape_data, columns=shape_columns)\n",
    "\n",
    "print(f'Shape features extracted: {len(shape_df)} samples')\n",
    "shape_df.head()"
]))

# ── Combine Features ──
cells.append(md(["### 4.4 Combine All Features"]))
cells.append(code([
    "# Drop duplicate label columns from texture and shape\n",
    "texture_df_no_label = texture_df.drop(columns=['label'])\n",
    "shape_df_no_label = shape_df.drop(columns=['label'])\n",
    "\n",
    "# Combine all features\n",
    "final_df = pd.concat(\n",
    "    [df.drop(columns=['label']),\n",
    "     texture_df_no_label,\n",
    "     shape_df_no_label,\n",
    "     df['label']],\n",
    "    axis=1\n",
    ")\n",
    "\n",
    "print('Total samples:', len(final_df))\n",
    "print('Total features (including label):', final_df.shape[1])\n",
    "print()\n",
    "print('Class distribution:')\n",
    "print(final_df['label'].value_counts().rename({0: 'authentic', 1: 'counterfeit'}))\n",
    "print()\n",
    "final_df.head()"
]))

# ── Train-Test Split + Scaling ──
cells.append(md([
    "## 5. Data Preparation\n",
    "\n",
    "- 80/20 train-test split with stratification\n",
    "- StandardScaler normalization"
]))
cells.append(code([
    "# Separate features and label\n",
    "X = final_df.drop(columns=['label'])\n",
    "y = final_df['label']\n",
    "\n",
    "# Train-test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "# Normalize features\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "print('Training samples:', X_train_scaled.shape)\n",
    "print('Testing samples:', X_test_scaled.shape)"
]))

# ── Random Forest ──
cells.append(md([
    "## 6. Model Training & Evaluation\n",
    "\n",
    "### 6.1 Random Forest (RF)\n",
    "\n",
    "**Gini Index** (Paper Eq. 1):  \n",
    "Gini = 1 - [(B+)^2 + (B-)^2]"
]))
cells.append(code([
    "rf_model = RandomForestClassifier(\n",
    "    n_estimators=200,\n",
    "    max_depth=None,\n",
    "    class_weight='balanced',\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "rf_model.fit(X_train_scaled, y_train)\n",
    "y_pred_rf = rf_model.predict(X_test_scaled)\n",
    "\n",
    "print('=' * 60)\n",
    "print('RANDOM FOREST — Results')\n",
    "print('=' * 60)\n",
    "print('Accuracy:', accuracy_score(y_test, y_pred_rf))\n",
    "print()\n",
    "print(classification_report(y_test, y_pred_rf,\n",
    "                            target_names=['authentic', 'counterfeit']))\n",
    "print('Confusion Matrix:')\n",
    "print(confusion_matrix(y_test, y_pred_rf))"
]))

# ── SVM ──
cells.append(md([
    "### 6.2 Support Vector Machine (SVM)\n",
    "\n",
    "**SVM Decision Function** (Paper Eq. 3):  \n",
    "h(p) = q + sum(y_t * b_t * K(p, p_t))\n",
    "\n",
    "Using RBF kernel for nonlinear classification."
]))
cells.append(code([
    "svm_model = SVC(\n",
    "    kernel='rbf',\n",
    "    C=10,\n",
    "    gamma='scale',\n",
    "    class_weight='balanced',\n",
    "    probability=True,\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "svm_model.fit(X_train_scaled, y_train)\n",
    "y_pred_svm = svm_model.predict(X_test_scaled)\n",
    "\n",
    "print('=' * 60)\n",
    "print('SVM (RBF Kernel) — Results')\n",
    "print('=' * 60)\n",
    "print('Accuracy:', accuracy_score(y_test, y_pred_svm))\n",
    "print()\n",
    "print(classification_report(y_test, y_pred_svm,\n",
    "                            target_names=['authentic', 'counterfeit']))\n",
    "print('Confusion Matrix:')\n",
    "print(confusion_matrix(y_test, y_pred_svm))"
]))

# ── XGBoost ──
cells.append(md(["### 6.3 XGBoost Classifier"]))
cells.append(code([
    "xgb_model = XGBClassifier(\n",
    "    n_estimators=300,\n",
    "    max_depth=6,\n",
    "    learning_rate=0.05,\n",
    "    scale_pos_weight=6,  # imbalance handling\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "xgb_model.fit(X_train_scaled, y_train)\n",
    "y_pred_xgb = xgb_model.predict(X_test_scaled)\n",
    "\n",
    "print('=' * 60)\n",
    "print('XGBoost — Results')\n",
    "print('=' * 60)\n",
    "print('Accuracy:', accuracy_score(y_test, y_pred_xgb))\n",
    "print()\n",
    "print(classification_report(y_test, y_pred_xgb,\n",
    "                            target_names=['authentic', 'counterfeit']))\n",
    "print('Confusion Matrix:')\n",
    "print(confusion_matrix(y_test, y_pred_xgb))"
]))

# ── Comparison Table ──
cells.append(md([
    "## 7. Model Comparison\n",
    "\n",
    "Comparing all classifiers (matching Paper Table 2 and Graph 1)."
]))
cells.append(code([
    "models = {\n",
    "    'Random Forest': y_pred_rf,\n",
    "    'SVM (RBF)':     y_pred_svm,\n",
    "    'XGBoost':       y_pred_xgb\n",
    "}\n",
    "\n",
    "results = {}\n",
    "for name, y_pred in models.items():\n",
    "    results[name] = {\n",
    "        'Accuracy':  accuracy_score(y_test, y_pred),\n",
    "        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),\n",
    "        'Recall':    recall_score(y_test, y_pred, average='weighted', zero_division=0),\n",
    "        'F1-Score':  f1_score(y_test, y_pred, average='weighted', zero_division=0)\n",
    "    }\n",
    "\n",
    "comparison_df = pd.DataFrame(results).T\n",
    "comparison_df.index.name = 'Algorithm'\n",
    "\n",
    "print('=' * 70)\n",
    "print('          COMPARATIVE RESULTS — ALL MODELS')\n",
    "print('=' * 70)\n",
    "print(comparison_df.round(4).to_string())\n",
    "print()\n",
    "comparison_df.style.format('{:.2%}').set_caption('Model Performance Comparison').background_gradient(cmap='Greens', axis=0)"
]))

# ── Bar Chart ──
cells.append(code([
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "\n",
    "comparison_pct = comparison_df * 100\n",
    "comparison_pct.plot(kind='bar', ax=ax, width=0.7, edgecolor='black', linewidth=0.5)\n",
    "\n",
    "ax.set_title('Counterfeit Medicine Detection - Model Comparison',\n",
    "             fontsize=14, fontweight='bold')\n",
    "ax.set_ylabel('Score (%)', fontsize=12)\n",
    "ax.set_xlabel('Algorithm', fontsize=12)\n",
    "ax.set_ylim(0, 105)\n",
    "ax.legend(loc='lower right', fontsize=10)\n",
    "ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')\n",
    "\n",
    "for container in ax.containers:\n",
    "    ax.bar_label(container, fmt='%.1f%%', fontsize=8, padding=2)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ── Confusion Matrices ──
cells.append(md(["### Confusion Matrices"]))
cells.append(code([
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "\n",
    "for ax, (name, y_pred) in zip(axes, models.items()):\n",
    "    cm = confusion_matrix(y_test, y_pred)\n",
    "    disp = ConfusionMatrixDisplay(cm, display_labels=['Authentic', 'Counterfeit'])\n",
    "    disp.plot(ax=ax, cmap='Blues', colorbar=False)\n",
    "    ax.set_title(name, fontsize=13, fontweight='bold')\n",
    "\n",
    "plt.suptitle('Confusion Matrices', fontsize=15, fontweight='bold', y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ── Feature Importance ──
cells.append(md(["## 8. Feature Importance (Random Forest)"]))
cells.append(code([
    "feature_names = [\n",
    "    'mean_R', 'mean_G', 'mean_B', 'std_R', 'std_G', 'std_B',\n",
    "    'contrast', 'energy', 'homogeneity', 'correlation',\n",
    "    'area', 'perimeter', 'aspect_ratio'\n",
    "]\n",
    "\n",
    "importances = rf_model.feature_importances_\n",
    "indices = np.argsort(importances)[::-1]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 5))\n",
    "colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))\n",
    "ax.bar(range(len(feature_names)), importances[indices], color=colors)\n",
    "ax.set_xticks(range(len(feature_names)))\n",
    "ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')\n",
    "ax.set_title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')\n",
    "ax.set_ylabel('Importance')\n",
    "plt.tight_layout()\n",
    "plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ── Save Models ──
cells.append(md(["## 9. Save Trained Models"]))
cells.append(code([
    "joblib.dump(rf_model,  'rf_model.pkl')\n",
    "joblib.dump(svm_model, 'svm_model.pkl')\n",
    "joblib.dump(xgb_model, 'xgb_model.pkl')\n",
    "joblib.dump(scaler,    'scaler.pkl')\n",
    "\n",
    "print('Models saved:')\n",
    "print('  - rf_model.pkl')\n",
    "print('  - svm_model.pkl')\n",
    "print('  - xgb_model.pkl')\n",
    "print('  - scaler.pkl')"
]))

# ── Inference Function — EXACT COPY of original working code ──
cells.append(md([
    "## 10. Single-Image Inference Pipeline\n",
    "\n",
    "**Exactly matching the original working inference code:**  \n",
    "Image -> YOLO detection -> Crop first box -> Extract 13 features -> Scale -> XGBoost predict (threshold=0.30)"
]))
cells.append(code([
    "# Feature extraction function — EXACT match to original working code\n",
    "def extract_features(img):\n",
    "    # ----- COLOR FEATURES -----\n",
    "    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "    mean = np.mean(img_rgb, axis=(0,1))\n",
    "    std = np.std(img_rgb, axis=(0,1))\n",
    "\n",
    "    # ----- TEXTURE FEATURES -----\n",
    "    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n",
    "    gray = cv2.resize(gray, (256, 256))\n",
    "\n",
    "    glcm = graycomatrix(gray,\n",
    "                        distances=[1],\n",
    "                        angles=[0],\n",
    "                        levels=256,\n",
    "                        symmetric=True,\n",
    "                        normed=True)\n",
    "\n",
    "    contrast = graycoprops(glcm, 'contrast')[0, 0]\n",
    "    energy = graycoprops(glcm, 'energy')[0, 0]\n",
    "    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]\n",
    "    correlation = graycoprops(glcm, 'correlation')[0, 0]\n",
    "\n",
    "    # ----- SHAPE FEATURES -----\n",
    "    _, thresh = cv2.threshold(gray, 127, 255,\n",
    "                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)\n",
    "\n",
    "    contours, _ = cv2.findContours(thresh,\n",
    "                                   cv2.RETR_EXTERNAL,\n",
    "                                   cv2.CHAIN_APPROX_SIMPLE)\n",
    "\n",
    "    if len(contours) > 0:\n",
    "        cnt = max(contours, key=cv2.contourArea)\n",
    "        area = cv2.contourArea(cnt)\n",
    "        perimeter = cv2.arcLength(cnt, True)\n",
    "        x, y, w, h = cv2.boundingRect(cnt)\n",
    "        aspect_ratio = float(w) / h if h != 0 else 0\n",
    "    else:\n",
    "        area = 0\n",
    "        perimeter = 0\n",
    "        aspect_ratio = 0\n",
    "\n",
    "    features = [\n",
    "        mean[0], mean[1], mean[2],\n",
    "        std[0], std[1], std[2],\n",
    "        contrast, energy, homogeneity, correlation,\n",
    "        area, perimeter, aspect_ratio\n",
    "    ]\n",
    "\n",
    "    return np.array(features).reshape(1, -1)\n",
    "\n",
    "\n",
    "# Predict function — EXACT match to original working code\n",
    "def predict_image(image_path, model=None, model_name='XGBoost', threshold=0.30):\n",
    "    if model is None:\n",
    "        model = xgb_model\n",
    "\n",
    "    # Step 1: YOLO detection\n",
    "    results = yolo_model(image_path, verbose=False)\n",
    "\n",
    "    if len(results[0].boxes) == 0:\n",
    "        print('No medicine detected.')\n",
    "        return None\n",
    "\n",
    "    # Take first detected box\n",
    "    box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)\n",
    "\n",
    "    img = cv2.imread(image_path)\n",
    "    crop = img[box[1]:box[3], box[0]:box[2]]\n",
    "\n",
    "    # Step 2: Extract features\n",
    "    features = extract_features(crop)\n",
    "\n",
    "    # Step 3: Scale features\n",
    "    feat_names = [\n",
    "        'mean_R', 'mean_G', 'mean_B',\n",
    "        'std_R', 'std_G', 'std_B',\n",
    "        'contrast', 'energy', 'homogeneity', 'correlation',\n",
    "        'area', 'perimeter', 'aspect_ratio'\n",
    "    ]\n",
    "\n",
    "    features_df = pd.DataFrame(features, columns=feat_names)\n",
    "    features_scaled = scaler.transform(features_df)\n",
    "\n",
    "    # Step 4: Predict\n",
    "    if hasattr(model, 'predict_proba'):\n",
    "        prob = model.predict_proba(features_scaled)[0]\n",
    "        prediction = 1 if prob[1] > threshold else 0\n",
    "        confidence = prob[prediction]\n",
    "    else:\n",
    "        prediction = model.predict(features_scaled)[0]\n",
    "        confidence = None\n",
    "\n",
    "    label = 'AUTHENTIC' if prediction == 0 else 'COUNTERFEIT'\n",
    "\n",
    "    if confidence is not None:\n",
    "        print(f'[{model_name}] Prediction: {label}  (confidence: {confidence:.4f})')\n",
    "    else:\n",
    "        print(f'[{model_name}] Prediction: {label}')\n",
    "\n",
    "    return prediction\n",
    "\n",
    "print('Inference functions defined.')"
]))

# ── ind_test Evaluation ──
cells.append(md([
    "## 11. Independent Testing (ind_test/)\n",
    "\n",
    "Test all 3 models on the 10 test images."
]))
cells.append(code([
    "# Ground truth\n",
    "ind_test_gt = {}\n",
    "for i in range(1, 6):\n",
    "    ind_test_gt[f'fake{i}.jpg'] = {'label': 1, 'expected': 'COUNTERFEIT'}\n",
    "    ind_test_gt[f'org{i}.jpg']  = {'label': 0, 'expected': 'AUTHENTIC'}\n",
    "\n",
    "ind_test_dir = 'ind_test'\n",
    "all_results = []\n",
    "\n",
    "# Models to test with their thresholds\n",
    "test_models = [\n",
    "    ('Random Forest', rf_model,  0.30),\n",
    "    ('SVM',           svm_model, 0.30),\n",
    "    ('XGBoost',       xgb_model, 0.30),\n",
    "]\n",
    "\n",
    "print('=' * 80)\n",
    "print('INDEPENDENT TESTING — ind_test/')\n",
    "print('=' * 80)\n",
    "\n",
    "for img_file in sorted(ind_test_gt.keys()):\n",
    "    img_path = os.path.join(ind_test_dir, img_file)\n",
    "    gt = ind_test_gt[img_file]\n",
    "\n",
    "    print(f'\\n--- {img_file} (expected: {gt[\"expected\"]}) ---')\n",
    "\n",
    "    row = {'Image': img_file, 'Expected': gt['expected']}\n",
    "\n",
    "    for model_name, model_obj, thresh in test_models:\n",
    "        pred = predict_image(img_path, model=model_obj,\n",
    "                            model_name=model_name, threshold=thresh)\n",
    "        if pred is not None:\n",
    "            pred_label = 'AUTHENTIC' if pred == 0 else 'COUNTERFEIT'\n",
    "            correct = 'OK' if pred == gt['label'] else 'WRONG'\n",
    "        else:\n",
    "            pred_label = 'NO_DETECT'\n",
    "            correct = 'WRONG'\n",
    "        row[model_name] = pred_label\n",
    "        row[f'{model_name}_correct'] = correct\n",
    "\n",
    "    all_results.append(row)"
]))

# ── Results Summary ──
cells.append(code([
    "results_df = pd.DataFrame(all_results)\n",
    "\n",
    "display_cols = ['Image', 'Expected', 'Random Forest', 'SVM', 'XGBoost']\n",
    "print('\\n' + '=' * 80)\n",
    "print('PREDICTION RESULTS — ind_test/')\n",
    "print('=' * 80)\n",
    "print(results_df[display_cols].to_string(index=False))\n",
    "\n",
    "print('\\n' + '=' * 80)\n",
    "print('ACCURACY on ind_test (10 images)')\n",
    "print('=' * 80)\n",
    "for model_name in ['Random Forest', 'SVM', 'XGBoost']:\n",
    "    col = f'{model_name}_correct'\n",
    "    n_correct = sum(results_df[col] == 'OK')\n",
    "    total = len(results_df)\n",
    "    print(f'  {model_name:>15}: {n_correct}/{total} correct ({n_correct/total*100:.0f}%)')"
]))

# ── Visual Grid ──
cells.append(md(["### Visual Results"]))
cells.append(code([
    "fig, axes = plt.subplots(2, 5, figsize=(20, 8))\n",
    "\n",
    "for idx, img_file in enumerate(sorted(ind_test_gt.keys())):\n",
    "    row_idx = idx // 5\n",
    "    col_idx = idx % 5\n",
    "    ax = axes[row_idx, col_idx]\n",
    "\n",
    "    img_path = os.path.join(ind_test_dir, img_file)\n",
    "    img = cv2.imread(img_path)\n",
    "    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "    ax.imshow(img_rgb)\n",
    "\n",
    "    gt = ind_test_gt[img_file]\n",
    "    res_row = [r for r in all_results if r['Image'] == img_file]\n",
    "    if res_row:\n",
    "        xgb_pred = res_row[0].get('XGBoost', '?')\n",
    "        correct = res_row[0].get('XGBoost_correct', 'WRONG')\n",
    "        color = 'green' if correct == 'OK' else 'red'\n",
    "        ax.set_title(f'{img_file}\\nExp: {gt[\"expected\"]}\\nPred: {xgb_pred}',\n",
    "                     fontsize=9, color=color, fontweight='bold')\n",
    "    ax.axis('off')\n",
    "\n",
    "plt.suptitle('Independent Test Results (XGBoost)', fontsize=16, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.savefig('ind_test_results.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ── Conclusion ──
cells.append(md([
    "## 12. Conclusion\n",
    "\n",
    "This notebook implements the pipeline from the research paper:  \n",
    "*\"Analyzing the Counterfeit Medicines Based on Classification Using Machine Learning Techniques\"*  \n",
    "by Binitha S. Thomson and W. Rose Varuna.\n",
    "\n",
    "**Pipeline:** `Input Image -> YOLOv8 Detection -> Crop -> Color/Texture/Shape Features -> StandardScaler -> RF / SVM / XGBoost -> Authentic / Counterfeit`\n",
    "\n",
    "**Key findings:**\n",
    "- Three classifiers (RF, SVM, XGBoost) were compared on hand-crafted features\n",
    "- The paper reported SVM achieving 94.5% accuracy\n",
    "- All models use class_weight='balanced' or scale_pos_weight to handle imbalance\n",
    "- XGBoost with threshold=0.30 provides the best real-world performance"
]))

# ── Build notebook ──
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("4.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to 4.ipynb")
print(f"Total cells: {len(cells)}")
print(f"  Markdown: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code:     {sum(1 for c in cells if c['cell_type'] == 'code')}")
