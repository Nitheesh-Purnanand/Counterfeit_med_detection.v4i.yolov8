"""Medicine Search Page — Look up any medicine from the 256K database."""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import CUSTOM_CSS, _load_med_db

st.set_page_config(page_title="Medicine Search", page_icon="💊",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading medicine database...")
def get_db(): return _load_med_db()
med_db = get_db()

with st.sidebar:
    st.caption("Search across 256,000+ medicines")

st.markdown('<div class="hero-title">Medicine Search</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Search any medicine to find uses, side effects, '
            'composition, substitutes and more</div>', unsafe_allow_html=True)

if med_db is None:
    st.error("Medicine database not found."); st.stop()

def _show_results(hits):
    st.caption(f"Showing {len(hits)} result(s)")
    for idx, (_, row) in enumerate(hits.iterrows()):
        name = str(row.get("name",""))
        mfr = str(row.get("manufacturer_name","")); mfr = "" if mfr=="nan" else mfr
        med_type = str(row.get("type","")).title(); med_type = "" if med_type=="Nan" else med_type
        price_col = [c for c in row.index if 'price' in c.lower()]
        price = row[price_col[0]] if price_col else "N/A"
        with st.expander(f"{name}", expanded=(idx==0)):
            k1,k2,k3 = st.columns(3)
            k1.markdown(f"""<div class="glass-card" style="text-align:center;">
                <div class="metric-value" style="color:#14967f; font-size:1.4rem;">Rs. {price}</div>
                <div class="metric-label">Price</div></div>""", unsafe_allow_html=True)
            k2.markdown(f"""<div class="glass-card" style="text-align:center;">
                <div style="font-size:0.85rem; font-weight:700; color:#095d7e;">{mfr}</div>
                <div class="metric-label">Manufacturer</div></div>""", unsafe_allow_html=True)
            pack = str(row.get("pack_size_label","")); pack = "" if pack=="nan" else pack
            k3.markdown(f"""<div class="glass-card" style="text-align:center;">
                <div style="font-size:0.85rem; font-weight:700; color:#14967f;">{med_type}</div>
                <div class="metric-label">{pack}</div></div>""", unsafe_allow_html=True)
            c1 = str(row.get("short_composition1","")).strip()
            c2 = str(row.get("short_composition2","")).strip()
            if c1 and c1!="nan":
                cs = c1 + (f" + {c2}" if c2 and c2!="nan" else "")
                st.markdown(f"""<div class="glass-card"><h4>Composition</h4>
                    <p>{cs}</p></div>""", unsafe_allow_html=True)
            uses = [str(row.get(f"use{i}","")).strip() for i in range(5)]
            uses = [u for u in uses if u and u!="nan"]
            if uses:
                uh = "".join(f'<li style="margin:3px 0;">{u}</li>' for u in uses)
                st.markdown(f"""<div class="glass-card"><h4>Uses</h4>
                    <ul>{uh}</ul></div>""", unsafe_allow_html=True)
            se = str(row.get("Consolidated_Side_Effects","")).strip()
            if se and se!="nan":
                st.markdown(f"""<div class="glass-card" style="border-left:3px solid #d63031;">
                    <h4>Side Effects</h4><p>{se}</p></div>""", unsafe_allow_html=True)
            subs = [str(row.get(f"substitute{i}","")).strip() for i in range(5)]
            subs = [s for s in subs if s and s!="nan"]
            if subs:
                sh = " ".join(f'<span class="tag">{s}</span>' for s in subs)
                st.markdown(f"""<div class="glass-card"><h4>Substitutes</h4>
                    <div>{sh}</div></div>""", unsafe_allow_html=True)
            extras = [("Chemical Class",),("Therapeutic Class",),
                      ("Action Class",),("Habit Forming",)]
            ei = [(c[0], str(row.get(c[0],"")).strip()) for c in extras]
            ei = [(c,v) for c,v in ei if v and v!="nan"]
            if ei:
                cols = st.columns(len(ei))
                for col,(c,v) in zip(cols,ei):
                    col.markdown(f"""<div class="glass-card" style="text-align:center; padding:0.6rem;">
                        <div style="font-size:0.75rem; font-weight:700; color:#095d7e;">{v}</div>
                        <div class="metric-label">{c}</div></div>""", unsafe_allow_html=True)
            disc = str(row.get("Is_discontinued","False")).strip()
            if disc.lower()=="true":
                st.warning("This medicine has been **discontinued**.")

# Search options
st.markdown("### Search Medicine")
all_meds = med_db["name"].dropna().drop_duplicates().tolist()
selected = st.selectbox("Search / Select medicine", [""] + all_meds)

if selected:
    _show_results(med_db[med_db["name"]==selected].head(1))
else:
    st.markdown("""<div class="glass-card" style="text-align:center; padding:2rem;">
        <h3>Select a medicine from the dropdown above</h3>
        <p>Access details for over 256,000 medicines.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""<div style="text-align:center; color:#888; font-size:0.8rem;">
    Dataset: Extensive A-Z Medicines Dataset of India (256,476 medicines)
</div>""", unsafe_allow_html=True)
