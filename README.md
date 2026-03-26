# 🛡️ DeepShield — Deepfake Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=flat-square&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?style=flat-square&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?style=flat-square&logo=streamlit)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?style=flat-square&logo=flask)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)

**An end-to-end deepfake image detection system combining a custom Lightweight CNN with FFT-based frequency domain analysis — deployed as a dark-themed Streamlit dashboard and a Flask REST API.**

---

##  Architecture Overview

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
│  Lightweight CNN   │     │  FFT Analysis         │
│  4 Conv Blocks     │     │  (frequency domain)   │
│  CNN Score: 0–1    │     │  FFT Score: 0–1       │
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
│   ├── model.py          ← Lightweight CNN architecture
│   ├── gradcam.py        ← Grad-CAM heatmap explainability
│   ├── fft_analysis.py   ← FFT frequency domain GAN artifact detection
│   └── train.py          ← Memory-safe training with ImageDataGenerator
├── utils/
│   ├── preprocess.py     ← Face detection pipeline (OpenCV + MTCNN)
│   └── helpers.py        ← Ensemble scoring, hints, result formatting
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
git clone https://github.com/YOUR_USERNAME/deepshield.git
cd deepshield
python -m venv venv
venv\Scripts\activate        # Windows
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
python -m model.train
```

Training uses `ImageDataGenerator` for memory-safe batch loading — no RAM overflow.

Outputs:
- `deepfake_efficientnet.h5` ← final trained model
- `deepfake_efficientnet_best.h5` ← best checkpoint
- `training_history.png` ← accuracy + loss curves

---

##  Running the Dashboard

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
| `model_used` | `"LightweightCNN"` |

Health check:
```bash
curl http://localhost:5000/health
# → {"status": "ok", "model": "LightweightCNN"}
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

Trained and evaluated on the [140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) dataset:

| Metric | Score |
|--------|-------|
| **Accuracy** | **93.70%** |
| **AUC** | **98.26%** |
| **Precision** | **92.50%** |
| **Recall** | **95.12%** |
| **Loss** | 0.2391 |

> Trained on 20,000 images (10K real + 10K fake) for 10 epochs using batch size 8 on CPU.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | TensorFlow 2.13, Keras |
| Model | Custom Lightweight CNN (4 Conv blocks, ~490K params) |
| Face Detection | OpenCV Haar Cascade + MTCNN |
| Frequency Analysis | NumPy FFT + SciPy |
| Explainability | Grad-CAM (custom implementation) |
| Dashboard | Streamlit 1.28 + Plotly |
| REST API | Flask 3.0 + Flask-CORS |
| Containerisation | Docker |
| Data Processing | scikit-learn, pandas |

---

## 💡 Key Technical Highlights

**1. Ensemble of two independent signals**
The system combines CNN spatial analysis with FFT frequency domain analysis. GAN and diffusion models leave periodic frequency artifacts invisible to the human eye — FFT detects these independently of the neural network, reducing false negatives.

**2. Explainable AI (XAI) with Grad-CAM**
Grad-CAM highlights exactly which facial regions triggered the FAKE decision — making the model interpretable rather than a black box. Critical for forensic use cases.

**3. Memory-safe training with ImageDataGenerator**
Images are loaded batch-by-batch from disk instead of all at once — enabling training on machines with limited RAM (tested on 8GB).

**4. Production-ready dual deployment**
Ships both a user-facing Streamlit dashboard and a machine-consumable REST API — same model, two interfaces, demonstrating full-stack system design thinking.

**5. Graceful degradation**
If no face is detected, the system analyses the full image instead of failing — important for real-world robustness.

---

## ⚠️ Known Limitations

- Lower accuracy on diffusion-model generated images (Midjourney, DALL-E) — model trained primarily on GAN faces
- CPU-only inference — predictions take 1–3 seconds per image
- No video support — frame-by-frame analysis would need to be added

---

## 🔗 Dataset Links

- [Real vs Fake Faces — Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics)
- [DFDC — Meta AI](https://ai.meta.com/datasets/dfdc/)

---

## 👤 Author

**Sanjana** · Final-year CSE · Campus Placement Project