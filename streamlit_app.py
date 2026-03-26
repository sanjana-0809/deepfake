"""
dashboard.py — DeepShield Professional Dashboard
Run: streamlit run app/dashboard.py
"""

import os, sys, time, datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.preprocess import preprocess_image
from utils.helpers import build_result, result_to_display_dict
from model.fft_analysis import compute_fft_score
from model.gradcam import generate_gradcam_overlay

st.set_page_config(
    page_title="DeepShield — Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    background-color: #080a0f; color: #cdd6f4;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #1e2433; }
.header-banner {
    background: linear-gradient(135deg, #0d1117 0%, #111827 50%, #0d1117 100%);
    border: 1px solid #1e2d40; border-radius: 16px;
    padding: 28px 36px; margin-bottom: 24px;
    display: flex; align-items: center; justify-content: space-between;
}
.header-left h1 { font-size: 2em; font-weight: 700; color: #e2e8f0; margin: 0 0 4px 0; letter-spacing: -0.5px; }
.header-left p  { color: #64748b; font-size: 0.88em; margin: 0; }
.header-stats   { display: flex; gap: 16px; }
.stat-pill { text-align: center; background: #0f172a; border: 1px solid #1e2d40; border-radius: 10px; padding: 10px 18px; }
.stat-pill .val { font-size: 1.25em; font-weight: 700; color: #38bdf8; display: block; }
.stat-pill .lbl { font-size: 0.68em; color: #475569; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stFileUploadDropzone"] {
    background-color: #0d1117 !important; border: 2px dashed #1e3a5f !important; border-radius: 14px !important;
}
.result-card { border-radius: 16px; padding: 22px; text-align: center; }
.result-card.fake { background: linear-gradient(145deg,#1a0a0e,#0f0d14); border: 1px solid #5c1d2a; }
.result-card.real { background: linear-gradient(145deg,#071612,#0d1117); border: 1px solid #0f4030; }
.verdict { font-size: 2.2em; font-weight: 800; letter-spacing: 3px; margin: 0 0 2px 0; }
.verdict.fake { color: #f87171; }
.verdict.real { color: #34d399; }
.verdict-icon { font-size: 1.6em; vertical-align: middle; margin-left: 8px; }
.conf-val { font-size: 1.9em; font-weight: 700; color: #94a3b8; margin: 4px 0 6px 0; }
.hint { font-size: 0.82em; color: #64748b; font-style: italic; margin-bottom: 18px; line-height: 1.5; }
.score-block { margin: 9px 0; }
.score-label-row { display: flex; justify-content: space-between; font-size: 0.78em; color: #94a3b8; margin-bottom: 4px; }
.score-label-row span:last-child { font-weight: 600; color: #cbd5e1; }
.bar-bg { background: #1e2433; border-radius: 4px; height: 5px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; }
.bar-cnn   { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
.bar-fft   { background: linear-gradient(90deg,#8b5cf6,#a78bfa); }
.bar-fake  { background: linear-gradient(90deg,#ef4444,#f87171); }
.bar-real  { background: linear-gradient(90deg,#10b981,#34d399); }
.thr-row { display:flex; justify-content:space-between; font-size:0.75em; color:#475569; margin-top:12px; padding-top:10px; border-top:1px solid #1e2433; }
.img-label { font-size:0.75em; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#475569; margin-bottom:7px; }
.face-warn { background:#1c1408; border:1px solid #3d2e08; border-radius:8px; padding:6px 12px; font-size:0.76em; color:#fbbf24; margin-top:7px; }
.model-card { background:#0f172a; border:1px solid #1e2d40; border-radius:12px; padding:14px 16px; margin:10px 0; }
.model-card .mname { font-size:0.83em; font-weight:600; color:#e2e8f0; margin-bottom:8px; }
.mrow { display:flex; justify-content:space-between; margin:4px 0; font-size:0.76em; }
.mrow .mk { color:#475569; } .mrow .mv { color:#38bdf8; font-weight:600; }
.step-item { display:flex; gap:9px; align-items:flex-start; margin:7px 0; font-size:0.8em; color:#64748b; line-height:1.5; }
.step-num { background:#0f172a; border:1px solid #1e2d40; border-radius:50%; width:19px; height:19px; min-width:19px; display:flex; align-items:center; justify-content:center; font-size:0.72em; color:#38bdf8; font-weight:600; }
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg,#3b82f6,#38bdf8) !important; }
.footer { text-align:center; color:#1e2433; font-size:0.73em; margin-top:36px; padding:14px 0; border-top:1px solid #0f172a; }
hr { border-color:#1e2433; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    from tensorflow.keras.models import load_model as km
    
    possible_paths = [
        "deepfake_efficientnet_best.h5",
        "deepfake_efficientnet.h5",
        "deepfake_cnn_model.h5",
        os.path.join(ROOT, "deepfake_efficientnet_best.h5"),
        os.path.join(ROOT, "deepfake_efficientnet.h5"),
    ]
    
    for path in possible_paths:
        if os.path.isfile(path):
            return km(path)
    
    return None


def analyse_image(pil_image, model, threshold):
    img_rgb = np.array(pil_image.convert("RGB"))
    img_array, face_found = preprocess_image(img_rgb, return_face_found=True)
    raw = float(model.predict(img_array, verbose=0)[0][0])
    cnn_score = 1.0 - raw
    fft_score = compute_fft_score(img_rgb)["fft_score"]
    result = build_result(cnn_score=cnn_score, fft_score=fft_score,
                          threshold=threshold, face_found=face_found)
    try:    gradcam_img = generate_gradcam_overlay(model, img_array, img_rgb)
    except: gradcam_img = img_rgb
    return result, gradcam_img, face_found


def make_gauge(score, label):
    c = "#f87171" if label == "FAKE" else "#34d399"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(score*100,1),
        number={"suffix":"%","font":{"color":c,"size":26}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#1e2433","tickfont":{"color":"#475569","size":9}},
               "bar":{"color":c,"thickness":0.22},"bgcolor":"#0d1117","bordercolor":"#1e2433",
               "steps":[{"range":[0,50],"color":"#0d1117"},{"range":[50,100],"color":"#0f172a"}],
               "threshold":{"line":{"color":"#64748b","width":1.5},"thickness":0.7,"value":50}}))
    fig.update_layout(height=180, margin=dict(t=8,b=0,l=8,r=8),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
    return fig


if "history" not in st.session_state:
    st.session_state.history = []

def add_to_history(name, result):
    st.session_state.history.append({
        "File": name, "Verdict": result["label"],
        "Confidence": result["confidence_percent"],
        "CNN": f"{result['cnn_score']:.3f}",
        "FFT": f"{result['fft_score']:.3f}",
        "Final": f"{result['final_score']:.3f}",
        "Time": datetime.datetime.now().strftime("%H:%M:%S"),
    })


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ DeepShield")
    st.caption("AI-Powered Deepfake Detection")
    st.divider()
    st.markdown("""
    <div class="model-card">
      <div class="mname">⚡ Lightweight CNN</div>
      <div class="mrow"><span class="mk">Accuracy</span><span class="mv">93.70%</span></div>
      <div class="mrow"><span class="mk">AUC</span><span class="mv">98.26%</span></div>
      <div class="mrow"><span class="mk">Precision</span><span class="mv">92.50%</span></div>
      <div class="mrow"><span class="mk">Recall</span><span class="mv">95.12%</span></div>
      <div class="mrow"><span class="mk">Parameters</span><span class="mv">490K · 1.87 MB</span></div>
    </div>
    """, unsafe_allow_html=True)
    threshold = st.slider("⚖️ Decision Threshold", 0.10, 0.90, 0.50, 0.05,
                          help="Score ≥ threshold → FAKE")
    st.divider()
    st.markdown("**How it works**")
    for n, t in [("1","Face detected via OpenCV + MTCNN"),
                 ("2","CNN scores manipulation probability"),
                 ("3","FFT detects GAN frequency artifacts"),
                 ("4","Ensemble = 70% CNN + 30% FFT"),
                 ("5","Grad-CAM shows suspicious regions")]:
        st.markdown(f'<div class="step-item"><div class="step-num">{n}</div><div>{t}</div></div>',
                    unsafe_allow_html=True)
    st.divider()
    st.caption("TensorFlow 2.13 · Lightweight CNN · Deepfake Detection · 2024")


# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading model …"):
    model = load_model()
if model is None:
    st.error("⚠️ No trained model found. Run `python -m model.train` first.", icon="🚨")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <div class="header-left">
    <h1>🛡️ DeepShield</h1>
    <p>AI-Powered Deepfake Detection &nbsp;·&nbsp; CNN + FFT Ensemble &nbsp;·&nbsp; Grad-CAM Explainability</p>
  </div>
  <div class="header-stats">
    <div class="stat-pill"><span class="val">93.7%</span><span class="lbl">Accuracy</span></div>
    <div class="stat-pill"><span class="val">98.3%</span><span class="lbl">AUC</span></div>
    <div class="stat-pill"><span class="val">95.1%</span><span class="lbl">Recall</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🖼️  Single Image", "📂  Batch Mode"])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader("Upload a face image — JPG, PNG or WEBP",
                                type=["jpg","jpeg","png","webp"])
    if uploaded:
        pil_image = Image.open(uploaded)
        bar = st.progress(0, text="Starting …")
        for p,t in [(15,"Detecting face …"),(40,"Running CNN …"),(70,"FFT analysis …")]:
            time.sleep(0.05); bar.progress(p, text=t)
        result, gradcam_img, face_found = analyse_image(pil_image, model, threshold)
        for p in [85,100]:
            time.sleep(0.03); bar.progress(p, text="Generating Grad-CAM …")
        bar.empty()
        add_to_history(uploaded.name, result)

        label    = result["label"]
        card_cls = "fake" if label=="FAKE" else "real"
        icon     = "❌" if label=="FAKE" else "✅"
        bar_cls  = "bar-fake" if label=="FAKE" else "bar-real"
        conf_val = result["final_score"] if label=="FAKE" else 1-result["final_score"]

        c1, c2, c3 = st.columns([1,1,1.05], gap="large")

        with c1:
            st.markdown('<div class="img-label">Original Image</div>', unsafe_allow_html=True)
            st.image(pil_image, use_column_width=True)
            if not face_found:
                st.markdown('<div class="face-warn">⚠️ No face found — full image used</div>',
                            unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="img-label">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
            st.image(gradcam_img, use_column_width=True)
            st.caption("🔴 Red = regions that influenced the prediction")

        with c3:
            cp = int(result["cnn_score"]*100)
            fp = int(result["fft_score"]*100)
            ep = int(result["final_score"]*100)
            st.markdown(f"""
            <div class="result-card {card_cls}">
              <div class="verdict {card_cls}">{label}<span class="verdict-icon">{icon}</span></div>
              <div class="conf-val">{result['confidence_percent']}</div>
              <div class="hint">{result['detection_hint']}</div>

              <div class="score-block">
                <div class="score-label-row"><span>CNN Score</span><span>{result['cnn_score']:.4f}</span></div>
                <div class="bar-bg"><div class="bar-fill bar-cnn" style="width:{cp}%"></div></div>
              </div>
              <div class="score-block">
                <div class="score-label-row"><span>FFT Score</span><span>{result['fft_score']:.4f}</span></div>
                <div class="bar-bg"><div class="bar-fill bar-fft" style="width:{fp}%"></div></div>
              </div>
              <div class="score-block">
                <div class="score-label-row"><span>Final Score</span><span>{result['final_score']:.4f}</span></div>
                <div class="bar-bg"><div class="bar-fill {bar_cls}" style="width:{ep}%"></div></div>
              </div>
              <div class="thr-row"><span>Threshold</span><span>{result['threshold_used']}</span></div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(make_gauge(conf_val, label), use_container_width=True)

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("Upload multiple images to analyse them all at once.")
    batch_files = st.file_uploader("Select images", type=["jpg","jpeg","png","webp"],
                                   accept_multiple_files=True,
                                   label_visibility="collapsed", key="batch")
    if batch_files:
        results = []
        prog = st.progress(0, text="Processing …")
        for i, f in enumerate(batch_files):
            prog.progress(int((i+1)/len(batch_files)*100),
                          text=f"Analysing {f.name} ({i+1}/{len(batch_files)}) …")
            res, _, _ = analyse_image(Image.open(f), model, threshold)
            res["image_name"] = f.name
            results.append(result_to_display_dict(res))
            add_to_history(f.name, res)
        prog.empty()

        df = pd.DataFrame(results)
        n_fake = (df["Label"]=="FAKE").sum()
        n_real = (df["Label"]=="REAL").sum()

        ca, cb, cc = st.columns(3)
        ca.metric("Total Images", len(df))
        cb.metric("Detected FAKE", int(n_fake))
        cc.metric("Detected REAL", int(n_real))
        st.markdown("---")

        def hl(v): return "color:#f87171;font-weight:600" if v=="FAKE" else "color:#34d399;font-weight:600"
        st.dataframe(df.style.applymap(hl, subset=["Label"]), hide_index=True)
        st.download_button("📥 Export as CSV", df.to_csv(index=False).encode(),
                           "deepshield_results.csv", "text/csv")

# ── Session History ───────────────────────────────────────────────────────────
st.divider()
st.markdown("#### 📋 Session History")
if st.session_state.history:
    hdf = pd.DataFrame(st.session_state.history)
    def hlv(v): return "color:#f87171;font-weight:600" if v=="FAKE" else "color:#34d399;font-weight:600"
    st.dataframe(hdf.style.applymap(hlv, subset=["Verdict"]), hide_index=True)
    col_a, _ = st.columns([1,5])
    with col_a:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []; st.rerun()
else:
    st.caption("No images analysed yet this session.")

st.markdown("""
<div class="footer">
  DeepShield &nbsp;·&nbsp; Built by Sanjana &nbsp;·&nbsp; TensorFlow 2.13 · OpenCV · Streamlit · 2024
</div>
""", unsafe_allow_html=True)