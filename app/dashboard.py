"""
dashboard.py — Professional Deepfake Detection Streamlit Dashboard

Run:
    streamlit run app/dashboard.py

Features:
  - Dark-themed UI
  - Sidebar: model selector, threshold slider, about section
  - Tab 1: Single image analysis with Grad-CAM, gauge chart, score breakdown
  - Tab 2: Batch upload with table + CSV export
  - Session history table at the bottom
"""

import os
import sys
import io
import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# Allow running from any directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.preprocess import preprocess_image
from utils.helpers import build_result, result_to_display_dict
from model.fft_analysis import compute_fft_score
from model.gradcam import generate_gradcam_overlay


# ──────────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DeepShield — Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark theme with accent colours
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Global ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d0f14;
        color: #e0e4f0;
        font-family: 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #141720;
        border-right: 1px solid #2a2e3e;
    }
    /* ── Headers ── */
    h1 { color: #7eb8f7; letter-spacing: -0.5px; }
    h2 { color: #a3c4f3; }
    h3 { color: #c5d8f8; }
    /* ── Upload zone ── */
    [data-testid="stFileUploadDropzone"] {
        background-color: #1a1f2e !important;
        border: 2px dashed #3a4060 !important;
        border-radius: 12px !important;
    }
    /* ── Result cards ── */
    .result-card {
        background: linear-gradient(135deg, #1a1f2e, #141720);
        border: 1px solid #2a2e3e;
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
    }
    .fake-badge {
        font-size: 2.8em;
        font-weight: 900;
        color: #ff4d6d;
        letter-spacing: 2px;
    }
    .real-badge {
        font-size: 2.8em;
        font-weight: 900;
        color: #2dd4bf;
        letter-spacing: 2px;
    }
    .confidence-text {
        font-size: 1.6em;
        font-weight: 700;
        color: #7eb8f7;
        margin: 6px 0;
    }
    .hint-text {
        font-size: 0.95em;
        color: #8892b0;
        font-style: italic;
        margin-top: 8px;
    }
    .score-row {
        display: flex;
        justify-content: space-between;
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1px solid #2a2e3e;
        font-size: 0.88em;
        color: #a8b2cc;
    }
    /* ── Metric chips ── */
    .chip {
        display: inline-block;
        background: #1e2535;
        border: 1px solid #303654;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.82em;
        color: #7eb8f7;
        margin: 2px;
    }
    /* ── Progress bar color ── */
    [data-testid="stProgress"] > div > div {
        background-color: #7eb8f7 !important;
    }
    /* ── Table ── */
    [data-testid="stDataFrame"] { border-radius: 8px; }
    /* ── Divider ── */
    hr { border-color: #2a2e3e; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(model_type: str):
    """Load the saved Keras model once and cache it for the session."""
    from tensorflow.keras.models import load_model as keras_load

    if model_type == "EfficientNetB4 (Recommended)":
        path = os.path.join(ROOT, "deepfake_efficientnet.h5")
        fallback = os.path.join(ROOT, "deepfake_efficientnet_best.h5")
    else:
        path = os.path.join(ROOT, "deepfake_cnn_model.h5")
        fallback = path

    if os.path.isfile(path):
        return keras_load(path)
    elif os.path.isfile(fallback):
        return keras_load(fallback)
    else:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis pipeline
# ──────────────────────────────────────────────────────────────────────────────

def analyse_image(pil_image: Image.Image, model, threshold: float) -> tuple:
    """
    Run the full analysis pipeline on a PIL image.

    Returns:
        result      : dict from helpers.build_result
        gradcam_img : uint8 RGB numpy array
        face_found  : bool
    """
    img_rgb = np.array(pil_image.convert("RGB"))

    # 1. Preprocess
    img_array, face_found = preprocess_image(img_rgb, return_face_found=True)

    # 2. CNN prediction
    cnn_score = float(model.predict(img_array, verbose=0)[0][0])

    # 3. FFT analysis
    fft_result = compute_fft_score(img_rgb)
    fft_score = fft_result["fft_score"]

    # 4. Build result dict
    result = build_result(
        cnn_score=cnn_score,
        fft_score=fft_score,
        threshold=threshold,
        face_found=face_found,
    )

    # 5. Grad-CAM overlay
    try:
        gradcam_img = generate_gradcam_overlay(model, img_array, img_rgb)
    except Exception:
        # If Grad-CAM fails (e.g. architecture mismatch), return the original
        gradcam_img = img_rgb

    return result, gradcam_img, face_found


# ──────────────────────────────────────────────────────────────────────────────
# Plotly gauge chart
# ──────────────────────────────────────────────────────────────────────────────

def make_gauge(score: float, label: str) -> go.Figure:
    colour = "#ff4d6d" if label == "FAKE" else "#2dd4bf"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(score * 100, 1),
            number={"suffix": "%", "font": {"color": colour, "size": 32}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#4a5278",
                         "tickfont": {"color": "#8892b0"}},
                "bar": {"color": colour},
                "bgcolor": "#1a1f2e",
                "bordercolor": "#2a2e3e",
                "steps": [
                    {"range": [0, 50],  "color": "#1a2530"},
                    {"range": [50, 65], "color": "#1e2a35"},
                    {"range": [65, 85], "color": "#22272e"},
                    {"range": [85, 100],"color": "#2a1a20"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(t=10, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e4f0",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Session history (stored in st.session_state)
# ──────────────────────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []


def add_to_history(name: str, result: dict):
    st.session_state.history.append({
        "Image Name": name,
        "Label": result["label"],
        "Confidence": result["confidence_percent"],
        "Threshold": result["threshold_used"],
        "Time": datetime.datetime.now().strftime("%H:%M:%S"),
    })


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ DeepShield")
    st.markdown("*AI-Powered Deepfake Detection*")
    st.divider()

    model_choice = st.selectbox(
        "🔬 Model",
        ["EfficientNetB4 (Recommended)", "Lightweight CNN"],
        index=0,
    )

    threshold = st.slider(
        "⚖️ Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Scores ≥ threshold → FAKE. Lower = more sensitive to deepfakes.",
    )

    st.divider()
    st.markdown("### ℹ️ How it works")
    st.markdown(
        """
        1. **Face detection** via OpenCV / MTCNN  
        2. **EfficientNetB4** predicts manipulation probability  
        3. **FFT analysis** detects GAN frequency artifacts  
        4. **Ensemble score** = 70% CNN + 30% FFT  
        5. **Grad-CAM** highlights suspicious regions  
        """
    )

    st.divider()
    st.caption("Built for campus placements · TensorFlow 2.13 · CPU-optimised")

# ──────────────────────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────────────────────

with st.spinner("Loading model …"):
    model = load_model(model_choice)

if model is None:
    st.error(
        "⚠️ No trained model found. "
        "Run `python model/train.py` first to train and save the model.",
        icon="🚨",
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Main title
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("# 🛡️ DeepShield — Deepfake Detection System")
st.markdown(
    "Upload a face image to detect whether it is **real** or **AI-generated / manipulated**."
)
st.divider()

tab_single, tab_batch = st.tabs(["🖼️ Single Image", "📂 Batch Mode"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Single Image
# ──────────────────────────────────────────────────────────────────────────────

with tab_single:
    uploaded = st.file_uploader(
        "Drag & drop or click to upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="visible",
    )

    if uploaded:
        pil_image = Image.open(uploaded)
        progress_bar = st.progress(0, text="Analysing …")

        # Simulate step-wise progress for UX
        for pct in range(0, 60, 20):
            time.sleep(0.05)
            progress_bar.progress(pct, text="Preprocessing image …")

        result, gradcam_img, face_found = analyse_image(pil_image, model, threshold)

        for pct in range(60, 101, 20):
            time.sleep(0.03)
            progress_bar.progress(min(pct, 100), text="Generating Grad-CAM …")

        progress_bar.empty()
        add_to_history(uploaded.name, result)

        # ── 3-column layout ────────────────────────────────────────────────
        col1, col2, col3 = st.columns([1, 1, 1.2], gap="medium")

        with col1:
            st.markdown("**Original Image**")
            st.image(pil_image, use_container_width=True)
            if not face_found:
                st.caption("⚠️ No face detected — full image analysed")

        with col2:
            st.markdown("**Grad-CAM Heatmap**")
            st.image(gradcam_img, use_container_width=True,
                     caption="Red = high attention")

        with col3:
            label = result["label"]
            badge_class = "fake-badge" if label == "FAKE" else "real-badge"
            badge_icon  = "❌" if label == "FAKE" else "✅"

            st.markdown(
                f"""
                <div class="result-card">
                  <div class="{badge_class}">{label} {badge_icon}</div>
                  <div class="confidence-text">{result['confidence_percent']}</div>
                  <div class="hint-text">{result['detection_hint']}</div>
                  <div class="score-row">
                    <span>CNN Score</span>
                    <span><b>{result['cnn_score']:.4f}</b></span>
                  </div>
                  <div class="score-row">
                    <span>FFT Score</span>
                    <span><b>{result['fft_score']:.4f}</b></span>
                  </div>
                  <div class="score-row">
                    <span>Final Score</span>
                    <span><b>{result['final_score']:.4f}</b></span>
                  </div>
                  <div class="score-row">
                    <span>Threshold</span>
                    <span><b>{result['threshold_used']}</b></span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            confidence_val = result["final_score"] if label == "FAKE" else (1 - result["final_score"])
            st.plotly_chart(
                make_gauge(confidence_val, label),
                use_container_width=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Batch Mode
# ──────────────────────────────────────────────────────────────────────────────

with tab_batch:
    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="visible",
        key="batch_uploader",
    )

    if batch_files:
        batch_results = []
        prog = st.progress(0, text="Processing batch …")

        for i, f in enumerate(batch_files):
            prog.progress(
                int((i + 1) / len(batch_files) * 100),
                text=f"Analysing {f.name} ({i+1}/{len(batch_files)}) …",
            )
            pil_img = Image.open(f)
            result, _, _ = analyse_image(pil_img, model, threshold)
            result["image_name"] = f.name
            batch_results.append(result_to_display_dict(result))
            add_to_history(f.name, result)

        prog.empty()

        df = pd.DataFrame(batch_results)

        # Colour-code Label column
        def highlight_label(val):
            if val == "FAKE":
                return "color: #ff4d6d; font-weight: bold"
            return "color: #2dd4bf; font-weight: bold"

        styled = df.style.applymap(highlight_label, subset=["Label"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── CSV export ─────────────────────────────────────────────────────
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export to CSV",
            data=csv_bytes,
            file_name="deepshield_batch_results.csv",
            mime="text/csv",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Session History
# ──────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("### 📋 Session History")

if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.caption("No images analysed yet in this session.")
