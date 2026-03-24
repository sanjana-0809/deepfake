# Dataset Download Instructions

This project supports the following deepfake detection datasets.
Place all data inside the `archive/` folder following the structure below.

---

## ✅ Default Dataset — Real vs Fake Faces (Kaggle)

The training script (`model/train.py`) uses this structure by default:

```
archive/
└── real_vs_fake/
    └── real-vs-fake/
        ├── train/
        │   ├── real/     ← real face images
        │   └── fake/     ← AI-generated face images
        ├── valid/
        │   ├── real/
        │   └── fake/
        └── test/
            ├── real/
            └── fake/
```

**Download link:**
https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

Size: ~3.2 GB | 140k images (70k real, 70k fake GAN-generated)

---

## 🔬 FaceForensics++ (FF++)

Academic benchmark — the gold standard for deepfake detection research.

**Access:** Requires filling a usage form (academic/research only).
https://github.com/ondyari/FaceForensics

Manipulations included:
- Deepfakes (DF)
- Face2Face (F2F)
- FaceShift (FS)
- NeuralTextures (NT)

Suggested folder mapping:
```
archive/ff++/
├── real/      ← original_sequences/youtube/
└── fake/      ← manipulated_sequences/Deepfakes/
```

---

## 🎬 Celeb-DF v2

High-quality deepfakes of celebrities — harder to detect than FF++.

**Download:** https://github.com/yuezunli/celeb-deepfakeforensics

Includes:
- 590 real YouTube videos of 59 celebrities
- 5,639 deepfake videos with improved synthesis

---

## 🏆 DFDC — DeepFake Detection Challenge

Facebook AI's large-scale dataset with 100k+ videos.

**Download:** https://ai.meta.com/datasets/dfdc/
(Requires Meta account and acceptance of terms)

---

## Quick Start (Kaggle CLI)

```bash
# Install Kaggle CLI
pip install kaggle

# Set up your API key (~/.kaggle/kaggle.json)
kaggle datasets download -d xhlulu/140k-real-and-fake-faces
unzip 140k-real-and-fake-faces.zip -d archive/
```

---

## Accuracy Benchmarks

| Dataset       | Accuracy | AUC  | Precision | Recall |
|---------------|----------|------|-----------|--------|
| Real vs Fake  | _____    | ____ | _____     | ____   |
| FF++ (c23)    | _____    | ____ | _____     | ____   |
| Celeb-DF v2   | _____    | ____ | _____     | ____   |

*Fill these in after training and evaluating your model.*
