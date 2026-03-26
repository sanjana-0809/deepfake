<div align="center">

# 🛡️ DeepShield
### Deepfake Detection System

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Click%20Here-00C853?style=for-the-badge)](https://deepfake-fraxrdqtzsajknmpfweh7m.streamlit.app/)

*An end-to-end deepfake image detection system using CNN + FFT + Grad-CAM*

</div>

---

##  Overview

**DeepShield** classifies images as **REAL or FAKE** using a hybrid approach combining:

| Component | Description |
|-----------|-------------|
|  **CNN** | Spatial feature extraction via EfficientNet |
|  **FFT Analysis** | Frequency-domain artifact detection |
|  **Grad-CAM** | Visual explainability heatmaps |

Deployed as a **dark-themed interactive Streamlit dashboard** with real-time predictions.

---

##  Architecture

```
Input Image
     │
     ▼
Face Detection (OpenCV + MTCNN)
     │
     ▼
CNN Model (Spatial Features) ────────┐
                                      ├──► Ensemble ──► Final Score
FFT Analysis (Frequency Domain) ─────┘
     │
     ▼
Threshold Decision (REAL / FAKE)
     │
     ▼
Grad-CAM Heatmap (Explainability)
```

---

##  Features

-  **Upload & Analyze** — Drag and drop any face image for instant prediction
-  **Demo Images** — Try preloaded sample images from the sidebar
-  **Real-Time Prediction** — Live inference with loading spinner
-  **Ensemble Scoring** — CNN + FFT scores combined for higher accuracy
-  **Grad-CAM Heatmap** — Visual explanation of model decisions
-  **Batch Processing** — Analyze multiple images at once
-  **Session History** — Track all predictions in a session
-  **CSV Export** — Download your results

---

##  Project Structure

```
deepfake/
├── streamlit_app.py          # Main application entry point
├── requirements.txt
├── runtime.txt
├── deepfake_efficientnet_best.h5   # Trained model weights
│
├── model/
│   ├── fft_analysis.py       # Frequency domain analysis
│   └── gradcam.py            # Grad-CAM heatmap generation
│
├── utils/
│   ├── preprocess.py         # Image preprocessing pipeline
│   └── helpers.py            # Utility functions
│
└── samples/
    ├── real1.jpg             # Sample real image
    └── fake1.jpg             # Sample deepfake image
```

---

##  Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/sanjana-0809/deepfake.git
cd deepfake

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run streamlit_app.py
```

---

##  How to Use

### 🔹 Upload an Image
1. Click **Browse files** in the main panel
2. Upload a face image (`.jpg`, `.png`)
3. View the verdict, confidence score, and heatmap

### 🔹 Use Demo Images
1. Open the **sidebar**
2. Click any sample button
3. Instantly test the model with preloaded images

---

##  Output

For each image, DeepShield provides:

| Output | Description |
|--------|-------------|
|   **Verdict** | REAL or FAKE classification |
|  **Confidence Score** | Overall prediction confidence |
|  **CNN Score** | Spatial feature analysis result |
|  **FFT Score** | Frequency domain analysis result |
|  **Grad-CAM Heatmap** | Visual explanation of the decision |

---

##  Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **93.7%** |
| AUC | **98.3%** |
| Precision | **92.5%** |
| Recall | **95.1%** |

---

##  Tech Stack

| Layer | Technology |
|-------|------------|
| **Deep Learning** | TensorFlow 2.13, Keras |
| **Model** | Lightweight CNN (~490K params) |
| **Face Detection** | OpenCV + MTCNN |
| **Frequency Analysis** | NumPy FFT |
| **Explainability** | Grad-CAM |
| **Dashboard** | Streamlit + Plotly |
| **Data Processing** | pandas, scikit-learn |

---

##  Key Highlights

-  **Hybrid CNN + FFT** approach improves detection accuracy over single-model baselines
-  **Grad-CAM** provides transparent, explainable AI insights into model decisions
-  **Lightweight model** — suitable for CPU deployment with no GPU required
-  **Fully deployed** — complete ML pipeline from Model → Backend → UI → Cloud

---

##  Limitations

- Works best on **clear, front-facing** face images
- Less accurate on **diffusion-generated** (e.g., Stable Diffusion) images
- **CPU-based inference** — slower than GPU-accelerated systems
- **No video detection** support yet

---

##  Future Improvements

-  Video deepfake detection
-  Mobile-friendly responsive UI
-  REST API deployment
-  Enhanced visualization & analytics dashboard

---

##  Author

<div align="center">

**Sanjana Ghatge**
*BTech in Computer Science Engineering*

[![GitHub](https://img.shields.io/badge/GitHub-sanjana--0809-181717?style=for-the-badge&logo=github)](https://github.com/sanjana-0809)
[![Email](https://img.shields.io/badge/Email-ghatgesanjana0809%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:ghatgesanjana0809@gmail.com)

</div>

---

<div align="center">

*This project demonstrates a complete end-to-end ML deployment pipeline:*

**Model → Backend → UI → Cloud**

 *If you found this useful, consider giving it a star!*

</div>