"""Shared utilities for the Counterfeit Medicine Detection app."""
import os, pickle, numpy as np, cv2, pandas as pd
from difflib import SequenceMatcher
from pathlib import Path

BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
PIPELINE_PKL  = os.path.join(BASE_DIR, "pipeline_data.pkl")
YOLO_WEIGHTS  = os.path.join(BASE_DIR, "runs", "detect", "train28", "weights", "best.pt")
YOLO_FALLBACK = os.path.join(BASE_DIR, "yolov8n.pt")
MODEL_CACHE   = os.path.join(BASE_DIR, "xgb_model_cache.pkl")
MED_DATASET   = os.path.join(BASE_DIR, "Extensive_A_Z_medicines_dataset_of_India.csv")
IMG_SIZE      = 128
CLASS_NAMES   = ["Authentic", "Counterfeit"]
FEATURE_GROUPS = [("Color Histogram",512),("LBP Texture",26),
                  ("GLCM Texture",5),("Hu Moments",7),("HOG Shape",1764)]

# ── CSS (shared) ──────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color-scheme: light; }
.stApp { background: #f1f9ff; color-scheme: light; }
.stApp, .stApp * { color-scheme: light; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #095d7e 0%, #14967f 100%) !important;
}
section[data-testid="stSidebar"] * { color: #e2fcd6 !important; }
section[data-testid="stSidebar"] .stMarkdown a { color: #e2fcd6 !important; }
h1, h2, h3, h4 { color: #095d7e !important; }
p, li, span, label, div { color: #333 !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div { color: #e2fcd6 !important; }

.hero-title {
    font-size: 2.2rem; font-weight: 800; color: #095d7e !important;
    margin-bottom: 0.2rem;
}
.hero-sub { font-size: 0.95rem; color: #555 !important; margin-bottom: 1.2rem; }
.glass-card {
    background: #ffffff !important; border: 1px solid #ccecee;
    border-radius: 14px; padding: 1.3rem;
    box-shadow: 0 4px 20px rgba(9,93,126,0.08); margin-bottom: 0.8rem;
}
.glass-card * { color: #333 !important; }
.glass-card h4 { color: #095d7e !important; }
.result-authentic {
    background: linear-gradient(135deg, #e2fcd6, #ccecee) !important;
    border: 2px solid #14967f; border-radius: 14px;
    padding: 1.8rem; text-align: center;
}
.result-counterfeit {
    background: linear-gradient(135deg, #ffe0e0, #fff0f0) !important;
    border: 2px solid #d63031; border-radius: 14px;
    padding: 1.8rem; text-align: center;
}
.metric-value { font-size: 2.2rem; font-weight: 800; }
.metric-label { font-size: 0.8rem; color: #666 !important; text-transform: uppercase; letter-spacing: 0.5px; }
.tag {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; margin: 2px;
    background: #ccecee; border: 1px solid #14967f; color: #095d7e !important;
}
.step-box {
    padding: 0.5rem 0.8rem; margin: 0.25rem 0; border-radius: 8px;
    background: rgba(255,255,255,0.3); border-left: 3px solid #e2fcd6;
    font-size: 0.82rem;
}
[data-testid="stFileUploadDropzone"], [data-testid="stFileUploaderDropzone"], [data-testid="stFileUploader"] > section {
    background-color: #ffffff !important;
}
[data-testid="stFileUploadDropzone"] *, [data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploader"] * {
    color: #333 !important;
}
[data-testid="stMainBlockContainer"] {
    background: #f1f9ff !important;
}
</style>
"""

# ── Feature extraction ────────────────────────────────────────────────────
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog

def extract_features(img_bgr):
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,[8,8,8],[0,180,0,256,0,256])
    f_color = cv2.normalize(hist, hist).flatten()
    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    f_lbp, _ = np.histogram(lbp.ravel(), bins=26, range=(0,26), density=True)
    q = (gray // 4).astype(np.uint8)
    glcm = graycomatrix(q,[1,3],[0,np.pi/4,np.pi/2],64,symmetric=True,normed=True)
    f_glcm = np.array([graycoprops(glcm,p).mean() for p in
                        ['contrast','dissimilarity','homogeneity','energy','correlation']])
    hu = cv2.HuMoments(cv2.moments(gray)).flatten()
    f_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    f_hog = hog(gray, orientations=9, pixels_per_cell=(16,16),
                cells_per_block=(2,2), feature_vector=True)
    return np.concatenate([f_color, f_lbp, f_glcm, f_hu, f_hog]).astype(np.float32)

# ── Loaders (call with @st.cache_resource in pages) ──────────────────────
def _load_pipeline():
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    if os.path.exists(MODEL_CACHE):
        with open(MODEL_CACHE, "rb") as f:
            return pickle.load(f)
    if not os.path.exists(PIPELINE_PKL):
        return None
    data = pickle.load(open(PIPELINE_PKL, "rb"))
    X_tr, X_te = data["X_train"], data["X_test"]
    y_tr, y_te = data["y_train"], data["y_test"]
    sc = data["scaler"]
    spw = np.sum(y_tr==0)/max(np.sum(y_tr==1),1)
    mdl = XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,
                         scale_pos_weight=spw,eval_metric="logloss",
                         random_state=42,n_jobs=-1,verbosity=0)
    mdl.fit(X_tr, y_tr)
    yp = mdl.predict(X_te)
    met = {"accuracy":accuracy_score(y_te,yp),
           "precision":precision_score(y_te,yp,pos_label=1),
           "recall":recall_score(y_te,yp,pos_label=1),
           "f1":f1_score(y_te,yp,pos_label=1),
           "cm":confusion_matrix(y_te,yp).tolist()}
    cache = {"model":mdl,"scaler":sc,"metrics":met}
    with open(MODEL_CACHE,"wb") as f: pickle.dump(cache,f)
    return cache

def _load_yolo():
    try:
        from ultralytics import YOLO
        w = YOLO_WEIGHTS if os.path.isfile(YOLO_WEIGHTS) else YOLO_FALLBACK
        if os.path.isfile(w): return YOLO(w)
    except Exception: pass
    return None

def _load_med_db():
    if not os.path.isfile(MED_DATASET): return None
    df = pd.read_csv(MED_DATASET, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["name_lower"] = df["name"].astype(str).str.lower().str.strip()
    return df

def _load_ocr():
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=True, verbose=False)
    except Exception:
        try:
            import easyocr
            return easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception: return None

def ocr_read(reader, img):
    if reader is None: return []
    try:
        return [(t.strip(),float(c)) for (_,t,c) in reader.readtext(img) if t.strip()]
    except Exception: return []

def fuzzy_match(query, db, threshold=0.45):
    if db is None or not query or len(query)<3: return None
    q = query.lower().strip()
    exact = db[db["name_lower"].str.contains(q, na=False, regex=False)]
    if len(exact)>0: return exact.iloc[0]
    cands = db[db["name_lower"].str.startswith(q[:2], na=False)].head(5000)
    if len(cands)==0: cands = db.head(10000)
    best_s, best_r = 0, None
    for _, r in cands.iterrows():
        s = SequenceMatcher(None, q, r["name_lower"]).ratio()
        if s > best_s: best_s, best_r = s, r
    return best_r if best_s >= threshold else None
