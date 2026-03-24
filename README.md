# 🛡️ DeepShield — Deepfake Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=flat-square&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?style=flat-square&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?style=flat-square&logo=streamlit)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?style=flat-square&logo=flask)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)

**An end-to-end deepfake image detection system combining EfficientNetB4 transfer learning with FFT-based frequency domain analysis — deployed as a dark-themed Streamlit dashboard and a Flask REST API.**

---

## 🧠 Architecture Overview

```
Input Image
     │
     ▼
┌────────────────────┐
│  Face Detection    │  OpenCV Haar Cascade → MTCNN fallback → full image
└────────┬───────────┘
         │  224×224 crop
         ▼
┌────────────────────┐     ┌──────────────────────┐
│  EfficientNetB4    │     │  FFT Analysis         │
│  + Custom Head     │     │  (frequency domain)   │
│  CNN Score: 0.0–1.0│     │  FFT Score: 0.0–1.0   │
└────────┬───────────┘     └──────────┬────────────┘
         │                            │
         └─────────────┬──────────────┘
                       │
               70% × CNN + 30% × FFT
                       │
                  Final Score
                       │
             ┌─────────▼─────────┐
             │  Threshold (0.5)  │
             └────────┬──────────┘
                      │
              FAKE ❌  or  REAL ✅
                      │
              Grad-CAM Overlay
```

---

## 📁 Project Structure

```
deepfake-detection-system/
├── model/
│   ├── model.py          ← EfficientNetB4 + lightweight CNN fallback
│   ├── gradcam.py        ← Grad-CAM heatmap explainability
│   ├── fft_analysis.py   ← FFT frequency domain GAN artifact detection
│   └── train.py          ← Two-phase training with callbacks
├── utils/
│   ├── preprocess.py     ← Face detection pipeline (OpenCV + MTCNN)
│   └── helpers.py        ← Scoring, hints, result formatting
├── app/
│   ├── dashboard.py      ← Streamlit dashboard (main UI)
│   └── api.py            ← Flask REST API
├── data/
│   └── download_data.md  ← Dataset download instructions
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourname/deepshield.git
cd deepshield
pip install -r requirements.txt
```

### 2. Download the dataset

See [data/download_data.md](data/download_data.md) for links and instructions.

Expected structure:
```
archive/real_vs_fake/real-vs-fake/
    train/real/  train/fake/
    valid/real/  valid/fake/
    test/real/   test/fake/
```

### 3. Train the model

```bash
python model/train.py
```

This runs:
- **Phase 1** — 10 epochs, head only (EfficientNetB4 base frozen)
- **Phase 2** — 5 epochs, top-30 layers unfrozen, lr=1e-5

Outputs:
- `deepfake_efficientnet.h5`   ← final model
- `deepfake_efficientnet_best.h5` ← best checkpoint
- `training_history.png`       ← accuracy + loss curves

---

## 🚀 Running the Dashboard

```bash
streamlit run app/dashboard.py
```

Open in browser: **http://localhost:8501**

Features:
- Upload single or batch images
- Grad-CAM heatmap overlay
- Plotly confidence gauge chart
- Batch CSV export
- Session history table

---

## 🔌 Running the Flask REST API

```bash
python app/api.py
```

API available at: **http://localhost:5000**

### Sample API request

```bash
curl -X POST http://localhost:5000/predict \
     -F "file=@/path/to/face.jpg"
```

### Sample API response

| Field | Value |
|-------|-------|
| `label` | `"FAKE"` |
| `confidence` | `"94.7%"` |
| `final_score` | `0.947` |
| `cnn_score` | `0.961` |
| `fft_score` | `0.901` |
| `detection_hint` | `"High confidence — Face swap or GAN artifact detected"` |
| `threshold_used` | `0.5` |
| `face_found` | `true` |
| `model_used` | `"EfficientNetB4"` |

Health check:
```bash
curl http://localhost:5000/health
# → {"status": "ok", "model": "EfficientNetB4"}
```

---

## 🐳 Docker

```bash
# Build
docker build -t deepshield .

# Run Streamlit dashboard
docker run -p 8501:8501 deepshield

# Run Flask API instead
docker run -p 5000:5000 deepshield python app/api.py
```

---

## 📊 Accuracy Benchmarks

| Dataset           | Accuracy | AUC  | Precision | Recall |
|-------------------|----------|------|-----------|--------|
| Real vs Fake Faces | ____%   | ____ | ____%     | ____%  |
| FaceForensics++ c23| ____%   | ____ | ____%     | ____%  |
| Celeb-DF v2       | ____%    | ____ | ____%     | ____%  |

*Fill in after training.*

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | TensorFlow 2.13, Keras |
| Backbone | EfficientNetB4 (ImageNet pretrained) |
| Face Detection | OpenCV Haar Cascade + MTCNN |
| Frequency Analysis | NumPy FFT + SciPy |
| Explainability | Grad-CAM (custom implementation) |
| Dashboard | Streamlit 1.28 + Plotly |
| REST API | Flask 3.0 + Flask-CORS |
| Containerisation | Docker |
| Data Processing | scikit-learn, pandas |

---

## 💡 Why This Is Powerful (For Interviews)

**1. Ensemble of two independent signals**
The system doesn't rely solely on a neural network. FFT analysis exploits the fact that GAN/diffusion models leave periodic frequency artifacts invisible to the human eye but detectable in the power spectrum — a completely different signal path that reduces false negatives.

**2. Explainable AI (XAI) with Grad-CAM**
Most deepfake detectors are black boxes. Grad-CAM shows *exactly* which facial regions triggered the decision — crucial for forensic use cases and demonstrates knowledge of modern XAI techniques.

**3. Two-phase transfer learning**
Freezing the backbone first then fine-tuning avoids catastrophic forgetting. This is industry-standard practice for adapting pretrained models to specialised domains.

**4. Production-ready deployment**
Ships both a user-facing Streamlit dashboard and a machine-consumable REST API — the same model behind two different interfaces, demonstrating system design thinking.

**5. Graceful degradation**
If no face is detected, the system still produces a result using the full image rather than failing — important for real-world robustness with varied inputs.

---

## 🔗 Dataset Links

- [Real vs Fake Faces — Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics)
- [DFDC — Meta AI](https://ai.meta.com/datasets/dfdc/)

---

## 👤 Author

**Sanjana** · Final-year CSE · Campus Placement Project
