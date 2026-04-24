"""Model Performance Page — Classifier comparison with confusion matrix."""
import streamlit as st, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import CUSTOM_CSS, CLASS_NAMES, _load_pipeline

st.set_page_config(page_title="Model Performance", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading pipeline...")
def get_pipe(): return _load_pipeline()

pipe = get_pipe()
if pipe is None: st.error("Pipeline not found."); st.stop()
metrics = pipe["metrics"]

with st.sidebar:
    st.caption("Model Performance Dashboard")

st.markdown('<div class="hero-title">Model Performance</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">XGBoost classifier evaluated on the held-out test set</div>',
            unsafe_allow_html=True)

# ── Key Metrics ───────────────────────────────────────────────────────────
st.markdown("### Key Metrics")
k1,k2,k3,k4 = st.columns(4)
for col, label, key, clr in [(k1,"Accuracy","accuracy","#14967f"),
                              (k2,"Precision","precision","#095d7e"),
                              (k3,"Recall","recall","#14967f"),
                              (k4,"F1 Score","f1","#095d7e")]:
    col.markdown(f"""<div class="glass-card" style="text-align:center;">
        <div class="metric-value" style="color:{clr};">{metrics[key]*100:.1f}%</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Confusion Matrix + Comparison ─────────────────────────────────────────
m1, m2 = st.columns(2)

with m1:
    st.markdown("### Confusion Matrix")
    cm = np.array(metrics["cm"])
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#f1f9ff"); ax.set_facecolor("#f1f9ff")
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                cbar_kws={"shrink": 0.8}, linewidths=1, linecolor="#ccecee")
    ax.set_ylabel("True Label", fontsize=10, color="#095d7e")
    ax.set_xlabel("Predicted Label", fontsize=10, color="#095d7e")
    ax.tick_params(colors="#333")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

with m2:
    st.markdown("### Classifier Comparison (Test Set)")
    st.markdown("""
    <div class="glass-card">
        <table style="width:100%; border-collapse:collapse; font-size:0.9rem;">
            <tr style="border-bottom:2px solid #ccecee;">
                <th style="text-align:left; padding:8px; color:#095d7e;">Model</th>
                <th style="text-align:right; padding:8px; color:#095d7e;">Accuracy</th>
                <th style="text-align:right; padding:8px; color:#095d7e;">Recall</th>
            </tr>
            <tr style="border-bottom:1px solid #eee;">
                <td style="padding:8px;">Random Forest</td>
                <td style="text-align:right; padding:8px;">94.1%</td>
                <td style="text-align:right; padding:8px;">83.3%</td>
            </tr>
            <tr style="border-bottom:1px solid #eee;">
                <td style="padding:8px;">Naive Bayes</td>
                <td style="text-align:right; padding:8px;">84.9%</td>
                <td style="text-align:right; padding:8px;">46.0%</td>
            </tr>
            <tr style="border-bottom:1px solid #eee;">
                <td style="padding:8px;">KNN</td>
                <td style="text-align:right; padding:8px;">91.8%</td>
                <td style="text-align:right; padding:8px;">75.7%</td>
            </tr>
            <tr style="border-bottom:1px solid #eee;">
                <td style="padding:8px;">SVM</td>
                <td style="text-align:right; padding:8px;">93.1%</td>
                <td style="text-align:right; padding:8px;">83.8%</td>
            </tr>
            <tr style="background:#e2fcd6; font-weight:700;">
                <td style="padding:8px; color:#095d7e;">XGBoost (Best)</td>
                <td style="text-align:right; padding:8px; color:#095d7e;">95.2%</td>
                <td style="text-align:right; padding:8px; color:#095d7e;">90.1%</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h4 style="color:#095d7e;">Why XGBoost?</h4>
        <ul>
            <li>Highest accuracy <b>(95.2%)</b> across all classifiers</li>
            <li>Best counterfeit recall <b>(90.1%)</b> &mdash; critical for safety</li>
            <li>Gradient boosting handles imbalanced classes effectively</li>
            <li>Combined with data augmentation for robust performance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div style="text-align:center; color:#888; font-size:0.8rem;">
    Base Paper: Thomson & Varuna (Springer, 2024) |
    Enhanced with YOLOv8, Data Augmentation, XGBoost & SHAP
</div>""", unsafe_allow_html=True)
