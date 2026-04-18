"""Generate pipeline_data.pkl with paths, features, labels, and trained models."""
import os, pickle, time, warnings
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
IMG_SIZE = 128
crops_dir = 'runs/detect/predict/crops'

# 1. Load all crops
image_paths = []
labels = []
for cls in ['authentic', 'counterfeit']:
    cls_dir = os.path.join(crops_dir, cls)
    lbl = 0 if cls == 'authentic' else 1
    for f in os.listdir(cls_dir):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_paths.append(os.path.join(cls_dir, f))
            labels.append(lbl)
labels = np.array(labels)
print(f'Total images: {len(image_paths)} (auth={np.sum(labels==0)}, fake={np.sum(labels==1)})')

# 2. Stratified split
paths_train, paths_test, y_train, y_test = train_test_split(
    image_paths, labels, test_size=0.20, random_state=42, stratify=labels)
print(f'Train: {len(paths_train)}, Test: {len(paths_test)}')

# 3. Feature extraction
def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color = cv2.calcHist([hsv],[0,1,2],None,[8,8,8],[0,180,0,256,0,256])
    color = cv2.normalize(color, color).flatten()
    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    lbp_h, _ = np.histogram(lbp.ravel(), bins=26, range=(0,26), density=True)
    img_q = (gray // 4).astype(np.uint8)
    glcm = graycomatrix(img_q, [1,3], [0, np.pi/4, np.pi/2], levels=64, symmetric=True, normed=True)
    glcm_f = np.array([graycoprops(glcm,p).mean() for p in ['contrast','dissimilarity','homogeneity','energy','correlation']])
    hu = cv2.HuMoments(cv2.moments(gray)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    hog_f = hog(gray, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), feature_vector=True)
    return np.concatenate([color, lbp_h, glcm_f, hu, hog_f])

def extract_batch(paths, name):
    feats, valid = [], []
    for i, p in enumerate(paths):
        if (i+1) % 500 == 0 or i == len(paths)-1:
            print(f'  {name}: {i+1}/{len(paths)}', end='\r')
        f = extract_features(p)
        if f is not None:
            feats.append(f); valid.append(i)
    print()
    return np.array(feats), valid

print('Extracting train features...')
X_train, tr_idx = extract_batch(paths_train, 'Train')
y_tr = y_train[tr_idx]
print('Extracting test features...')
X_test, te_idx = extract_batch(paths_test, 'Test')
y_te = y_test[te_idx]
# update paths to match valid indices
paths_train_valid = [paths_train[i] for i in tr_idx]
paths_test_valid = [paths_test[i] for i in te_idx]
print(f'Features: train={X_train.shape}, test={X_test.shape}')

# 4. Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# 5. Train models
scale_pos = np.sum(y_tr==0)/max(np.sum(y_tr==1),1)
classifiers = {
    'Random Forest (RF)': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    'Naive Bayes (NB)': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1),
    'SVM': SVC(kernel='rbf', class_weight='balanced', C=10, gamma='scale', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42, n_jobs=-1)
}

results = {}
trained_models = {}
for name, clf in classifiers.items():
    print(f'Training {name}...', end=' ')
    t0 = time.time()
    clf.fit(X_train_sc, y_tr)
    y_pred = clf.predict(X_test_sc)
    acc = accuracy_score(y_te, y_pred) * 100
    elapsed = time.time() - t0
    results[name] = {
        'accuracy': acc, 'time': elapsed, 'y_pred': y_pred,
        'report': classification_report(y_te, y_pred, target_names=['Authentic','Counterfeit']),
        'cm': confusion_matrix(y_te, y_pred)
    }
    trained_models[name] = clf
    print(f'{acc:.2f}% ({elapsed:.1f}s)')

# 6. Save everything
save_data = {
    'X_train': X_train, 'X_test': X_test,
    'y_tr': y_tr, 'y_te': y_te,
    'X_train_sc': X_train_sc, 'X_test_sc': X_test_sc,
    'paths_train': paths_train_valid, 'paths_test': paths_test_valid,
    'scaler': scaler, 'results': results,
    'trained_models': trained_models
}
pickle.dump(save_data, open('pipeline_data.pkl', 'wb'))
print(f'\nSaved pipeline_data.pkl ({os.path.getsize("pipeline_data.pkl")/1e6:.1f} MB)')
print('Done!')
