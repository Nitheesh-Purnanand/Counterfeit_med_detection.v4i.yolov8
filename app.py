"""Counterfeit Medicine Detection - Main Detection Page. Run: streamlit run app.py"""
import streamlit as st, numpy as np, cv2, matplotlib.pyplot as plt
from utils import (CUSTOM_CSS, CLASS_NAMES, FEATURE_GROUPS, IMG_SIZE,
                   extract_features, _load_pipeline, _load_yolo, _load_med_db,
                   _load_ocr, ocr_read, fuzzy_match)

st.set_page_config(page_title="Counterfeit Medicine Detector", page_icon="💊",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Cached loaders ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML pipeline...")
def get_pipeline(): return _load_pipeline()
@st.cache_resource(show_spinner="Loading YOLOv8...")
def get_yolo(): return _load_yolo()
@st.cache_resource(show_spinner="Loading medicine database...")
def get_med_db(): return _load_med_db()
@st.cache_resource(show_spinner="Loading OCR engine...")
def get_ocr(): return _load_ocr()

pipe = get_pipeline()
if pipe is None: st.error("Pipeline data not found. Run the notebook first."); st.stop()
model, scaler, metrics = pipe["model"], pipe["scaler"], pipe["metrics"]
yolo_model = get_yolo()
med_db = get_med_db()
ocr_reader = get_ocr()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.caption("Built with Streamlit | XGBoost | YOLOv8 | EasyOCR")

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Counterfeit Medicine Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload a medicine image &rarr; detection &rarr; '
            'classification &rarr; identification</div>', unsafe_allow_html=True)

# ── Helper: Medicine Info Card ────────────────────────────────────────────
def _show_medicine_info(m, pred_cls):
    if pred_cls == 1:
        st.markdown(f"""<div class="result-counterfeit" style="text-align:left;">
            <b>WARNING:</b> This medicine appears <b>counterfeit</b>. The text matched
            '{m["name"]}' in our database, but visual analysis indicates it may not be genuine.
            <b>Do not consume. Report to authorities.</b>
        </div>""", unsafe_allow_html=True)
        return

    st.markdown("---")
    st.markdown(f"### Medicine Details")
    st.caption(f'Identified as: **{m["name"]}**')

    ic1,ic2,ic3 = st.columns(3)
    pc = [c for c in m.index if 'price' in c.lower()]
    pv = m[pc[0]] if pc else 'N/A'
    mfr = str(m.get("manufacturer_name","N/A")); mfr = 'N/A' if mfr=='nan' else mfr
    mt = str(m.get("type","N/A")).title()
    pk = str(m.get("pack_size_label","N/A")); pk = 'N/A' if pk=='nan' else pk

    ic1.markdown(f"""<div class="glass-card" style="text-align:center;">
        <div class="metric-value" style="color:#14967f; font-size:1.6rem;">Rs. {pv}</div>
        <div class="metric-label">Price</div></div>""", unsafe_allow_html=True)
    ic2.markdown(f"""<div class="glass-card" style="text-align:center;">
        <div style="font-size:0.95rem; font-weight:700; color:#095d7e;">{mfr}</div>
        <div class="metric-label">Manufacturer</div></div>""", unsafe_allow_html=True)
    ic3.markdown(f"""<div class="glass-card" style="text-align:center;">
        <div style="font-size:0.95rem; font-weight:700; color:#14967f;">{mt}</div>
        <div class="metric-label">{pk}</div></div>""", unsafe_allow_html=True)

    c1 = str(m.get("short_composition1","")).strip()
    c2 = str(m.get("short_composition2","")).strip()
    if c1 and c1!='nan':
        cs = c1 + (f" + {c2}" if c2 and c2!='nan' else "")
        st.markdown(f"""<div class="glass-card"><h4>Composition</h4>
            <p style="font-size:1.05rem;">{cs}</p></div>""", unsafe_allow_html=True)

    uses = [str(m.get(f"use{i}","")).strip() for i in range(5)]
    uses = [u for u in uses if u and u!='nan']
    if uses:
        uh = "".join(f'<li style="margin:3px 0;">{u}</li>' for u in uses)
        st.markdown(f"""<div class="glass-card"><h4>Uses</h4>
            <ul>{uh}</ul></div>""", unsafe_allow_html=True)

    se = str(m.get("Consolidated_Side_Effects","")).strip()
    if se and se!='nan':
        st.markdown(f"""<div class="glass-card" style="border-left:3px solid #d63031;">
            <h4>Side Effects</h4><p>{se}</p></div>""", unsafe_allow_html=True)

    subs = [str(m.get(f"substitute{i}","")).strip() for i in range(5)]
    subs = [s for s in subs if s and s!='nan']
    if subs:
        sh = " ".join(f'<span class="tag">{s}</span>' for s in subs)
        st.markdown(f"""<div class="glass-card"><h4>Substitutes</h4>
            <div>{sh}</div></div>""", unsafe_allow_html=True)

    extras = [("Chemical Class",""),("Therapeutic Class",""),
              ("Action Class",""),("Habit Forming","")]
    ei = [(c,str(m.get(c,"")).strip()) for c,_ in extras]
    ei = [(c,v) for c,v in ei if v and v!='nan']
    if ei:
        cols = st.columns(len(ei))
        for col,(c,v) in zip(cols,ei):
            col.markdown(f"""<div class="glass-card" style="text-align:center; padding:0.8rem;">
                <div style="font-size:0.8rem; font-weight:700; color:#095d7e;">{v}</div>
                <div class="metric-label">{c}</div></div>""", unsafe_allow_html=True)

    disc = str(m.get("Is_discontinued","False")).strip()
    if disc.lower()=='true':
        st.warning("This medicine has been **discontinued**. Consult a pharmacist for alternatives.")


# ── Upload ────────────────────────────────────────────────────────────────
st.markdown("### Upload Medicine Image")
uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png","bmp","webp"],
                            label_visibility="collapsed")

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None: st.error("Could not read image."); st.stop()
    st.markdown("---")

    # ── YOLO Detection ────────────────────────────────────────────────────
    st.markdown("### Object Detection")
    crops = []
    if yolo_model is not None:
        results = yolo_model.predict(source=img_bgr, conf=0.25, verbose=False)
        annotated = img_bgr.copy()
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                h,w = img_bgr.shape[:2]
                x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(w,x2), min(h,y2)
                if x2>x1 and y2>y1:
                    crops.append(img_bgr[y1:y2, x1:x2])
                    cls_id = int(box.cls[0])
                    cls_label = "authentic" if cls_id==0 else "counterfeit"
                    # Blue boxes with filled label background (like reference)
                    blue = (255, 50, 0)  # BGR blue
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),blue,3)
                    # Label background
                    (tw,th),_ = cv2.getTextSize(cls_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(annotated,(x1,y1-th-10),(x1+tw+6,y1),blue,-1)
                    cv2.putText(annotated,cls_label,(x1+3,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                 caption=f"YOLOv8 Detection — {len(crops)} region(s) found",
                 use_container_width=True)
    else:
        st.info("YOLOv8 not available — using full image.")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width=400)
    if not crops: crops = [img_bgr]

    # ── Process each crop ─────────────────────────────────────────────────
    for ci, crop_img in enumerate(crops):
        if len(crops) > 1: st.markdown(f"---\n### Region {ci+1}")

        # ── Feature Extraction ────────────────────────────────────────────
        st.markdown("### Feature Extraction")
        feat_vec = extract_features(crop_img)
        # Show cropped region with red bounding box + label (like reference)
        disp = crop_img.copy()
        h_c, w_c = disp.shape[:2]
        red = (0, 0, 255)  # BGR red
        cv2.rectangle(disp, (2,2), (w_c-3, h_c-3), red, 3)
        lbl_text = "detected"
        (tw,th),_ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(disp, (2, 2), (2+tw+8, 2+th+10), red, -1)
        cv2.putText(disp, lbl_text, (6, 2+th+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        fc1, fc2 = st.columns([1, 2])
        fc1.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB),
                  caption="Detected Region", use_container_width=True)
        fc2.markdown(f"""<div class="glass-card">
            <h4>Feature Extraction Complete</h4>
            <p><b>{len(feat_vec)}</b> features extracted across 5 groups:</p>
            <div>{''.join(f'<span class="tag" style="margin:3px;">{n} ({d})</span>' for n,d in FEATURE_GROUPS)}</div>
        </div>""", unsafe_allow_html=True)

        # ── Classification ────────────────────────────────────────────────
        st.markdown("### Classification")
        feat_scaled = scaler.transform(feat_vec.reshape(1,-1))
        proba = model.predict_proba(feat_scaled)[0]
        pred_cls = int(np.argmax(proba)); conf = float(proba[pred_cls])
        css_cls = "result-authentic" if pred_cls==0 else "result-counterfeit"
        color = "#14967f" if pred_cls==0 else "#d63031"
        icon = "&#10003;" if pred_cls==0 else "&#10007;"

        r1,r2,r3 = st.columns([2,1,1])
        with r1:
            st.markdown(f"""<div class="{css_cls}">
                <div style="font-size:2.5rem; color:{color};">{icon}</div>
                <div class="metric-value" style="color:{color};">{CLASS_NAMES[pred_cls]}</div>
                <div class="metric-label">Prediction</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class="glass-card" style="text-align:center;">
                <div class="metric-value" style="color:{color};">{conf*100:.1f}%</div>
                <div class="metric-label">Confidence</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            risk = "LOW" if pred_cls==0 else ("HIGH" if conf>0.7 else "MEDIUM")
            rc = "#14967f" if risk=="LOW" else ("#d63031" if risk=="HIGH" else "#e17055")
            st.markdown(f"""<div class="glass-card" style="text-align:center;">
                <div class="metric-value" style="color:{rc};">{risk}</div>
                <div class="metric-label">Risk Level</div>
            </div>""", unsafe_allow_html=True)

        pb1,pb2 = st.columns(2)
        pb1.progress(float(proba[0]), text=f"Authentic: {proba[0]*100:.1f}%")
        pb2.progress(float(proba[1]), text=f"Counterfeit: {proba[1]*100:.1f}%")

        # ── Feature Importance ────────────────────────────────────────────
        st.markdown("### Explainability")
        imp = model.feature_importances_; off=0; gi={}
        for n,d in FEATURE_GROUPS: gi[n]=float(np.sum(imp[off:off+d])); off+=d
        fig,ax = plt.subplots(figsize=(7,2.5))
        fig.patch.set_facecolor("#f1f9ff"); ax.set_facecolor("#f1f9ff")
        ax.barh(list(gi.keys()), list(gi.values()),
                color=["#14967f","#095d7e","#e2fcd6","#ccecee","#14967f"],
                edgecolor="#095d7e", linewidth=0.5)
        ax.set_xlabel("Importance", fontsize=9)
        ax.set_title("Feature Group Importance", fontsize=11, fontweight="bold", color="#095d7e")
        ax.tick_params(labelsize=8)
        for s in ax.spines.values(): s.set_color("#ccecee")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        top_f = max(gi, key=gi.get)
        st.markdown(f"""<div class="glass-card">
            The model's decision relied most on <b>{top_f}</b> features,
            which capture {'color distribution' if 'Color' in top_f
            else 'structural/texture'} patterns of the medicine packaging.
        </div>""", unsafe_allow_html=True)

        # ── OCR + Medicine Info ───────────────────────────────────────────
        st.markdown("### Medicine Identification")
        med_match = None
        # Run OCR on FULL original image for better text detection
        ocr_texts = ocr_read(ocr_reader, img_bgr)
        if not ocr_texts:
            ocr_texts = ocr_read(ocr_reader, crop_img)
        if ocr_texts:
            st.markdown("**Detected text:**")
            for t,c in ocr_texts[:10]:
                st.markdown(f'<span class="tag">{t} ({c*100:.0f}%)</span>',
                            unsafe_allow_html=True)

        # Auto-match from OCR first
        if ocr_texts and med_db is not None:
            sorted_t = sorted(ocr_texts, key=lambda x: len(x[0]), reverse=True)
            for t,_ in sorted_t[:5]:
                med_match = fuzzy_match(t, med_db)
                if med_match is not None: break

        # Manual dropdown - search all (might be slow but avoids missing data)
        st.markdown("**Select medicine manually:**")
        if med_db is not None:
            all_meds = med_db["name"].dropna().drop_duplicates().tolist()
            manual_pick = st.selectbox("Search / Select medicine", [""] + all_meds, key=f"med_dd_{ci}")
            if manual_pick:
                med_match = med_db[med_db["name"]==manual_pick].iloc[0]

        # Display medicine info
        if med_match is not None:
            _show_medicine_info(med_match, pred_cls)

else:
    # Landing
    st.markdown("""<div class="glass-card" style="text-align:center; padding:2.5rem;">
        <div style="font-size:3rem; color:#14967f;">&#8682;</div>
        <h3>Upload a medicine image to get started</h3>
        <p>The system will detect, extract 2,314 features, classify, and identify the medicine.</p>
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div style="text-align:center; padding:0.5rem; color:#888; font-size:0.8rem;">
    <b>Counterfeit Medicine Detection</b> — Research Project |
    Base Paper: Thomson & Varuna (Springer, 2024) |
    Enhanced with YOLOv8, Data Augmentation, XGBoost & SHAP
</div>""", unsafe_allow_html=True)

