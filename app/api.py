"""
api.py — Flask REST API for Deepfake Detection

Endpoints:
    POST /predict
        Accepts a multipart image upload.
        Returns JSON with label, confidence, scores, and detection hint.

    GET /health
        Returns status and loaded model name.

Run:
    python app/api.py

CORS is enabled for all origins so the API can be consumed by any frontend.
"""

import os
import sys
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.preprocess import preprocess_image
from utils.helpers import build_result
from model.fft_analysis import compute_fft_score

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)   # allow cross-origin requests from any domain

# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

_model = None
_model_name = "none"


def _load_model():
    """
    Lazy-load the best available model once.
    Priority: deepfake_efficientnet.h5 → deepfake_efficientnet_best.h5 → deepfake_cnn_model.h5
    """
    global _model, _model_name

    if _model is not None:
        return _model

    from tensorflow.keras.models import load_model

    candidates = [
        ("deepfake_efficientnet.h5",      "EfficientNetB4"),
        ("deepfake_efficientnet_best.h5",  "EfficientNetB4 (best checkpoint)"),
        ("deepfake_cnn_model.h5",          "LightweightCNN"),
    ]

    for filename, name in candidates:
        path = os.path.join(ROOT, filename)
        if os.path.isfile(path):
            _model = load_model(path)
            _model_name = name
            print(f"[API] Loaded model: {name} ← {path}")
            return _model

    raise RuntimeError(
        "No trained model found. "
        "Run `python model/train.py` first to create deepfake_efficientnet.h5"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper: read uploaded image
# ──────────────────────────────────────────────────────────────────────────────

def _read_image_from_request() -> np.ndarray:
    """
    Read the uploaded file from request.files['file'] and return
    a uint8 RGB numpy array.

    Raises ValueError with a descriptive message on failure.
    """
    if "file" not in request.files:
        raise ValueError("No file part found. Send the image as 'file' in multipart/form-data.")

    f = request.files["file"]
    if f.filename == "":
        raise ValueError("Empty filename. Please attach a valid image file.")

    allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in allowed:
        raise ValueError(f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}")

    raw_bytes = f.read()
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    except Exception as e:
        raise ValueError(f"Could not decode image: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.

    Response:
        {
            "status": "ok",
            "model": "EfficientNetB4"
        }
    """
    try:
        _load_model()
        return jsonify({"status": "ok", "model": _model_name}), 200
    except RuntimeError as e:
        return jsonify({"status": "error", "detail": str(e)}), 503


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether an uploaded image is FAKE or REAL.

    Request:
        POST /predict
        Content-Type: multipart/form-data
        Body field: file  (image file)
        Optional query param: threshold  (float, default 0.5)

    Response 200:
        {
            "label":          "FAKE",
            "confidence":     "94.7%",
            "final_score":    0.947,
            "cnn_score":      0.961,
            "fft_score":      0.901,
            "detection_hint": "High confidence — Face swap or GAN artifact detected",
            "threshold_used": 0.5,
            "face_found":     true,
            "model_used":     "EfficientNetB4"
        }

    Response 400:
        { "error": "<description>" }

    Response 500:
        { "error": "<description>" }
    """
    # ── Parse threshold ────────────────────────────────────────────────────
    try:
        threshold = float(request.args.get("threshold", 0.5))
        if not 0.0 < threshold < 1.0:
            raise ValueError()
    except (ValueError, TypeError):
        return jsonify({"error": "threshold must be a float between 0.0 and 1.0."}), 400

    # ── Read image ─────────────────────────────────────────────────────────
    try:
        img_rgb = _read_image_from_request()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # ── Run inference ──────────────────────────────────────────────────────
    try:
        model = _load_model()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    try:
        img_array, face_found = preprocess_image(img_rgb, return_face_found=True)
        cnn_score = float(model.predict(img_array, verbose=0)[0][0])

        fft_result = compute_fft_score(img_rgb)
        fft_score = fft_result["fft_score"]

        result = build_result(
            cnn_score=cnn_score,
            fft_score=fft_score,
            threshold=threshold,
            image_name=request.files["file"].filename,
            face_found=face_found,
        )
    except Exception as e:
        return jsonify({"error": f"Inference error: {e}"}), 500

    # ── Build response ─────────────────────────────────────────────────────
    response_payload = {
        "label":          result["label"],
        "confidence":     result["confidence_percent"],
        "final_score":    result["final_score"],
        "cnn_score":      result["cnn_score"],
        "fft_score":      result["fft_score"],
        "detection_hint": result["detection_hint"],
        "threshold_used": result["threshold_used"],
        "face_found":     result["face_found"],
        "model_used":     _model_name,
    }

    return jsonify(response_payload), 200


# ──────────────────────────────────────────────────────────────────────────────
# Error handlers
# ──────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found. Available: POST /predict, GET /health"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed for this endpoint."}), 405


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    print("[API] Starting DeepShield Flask API on http://127.0.0.1:5000")
    print("[API] Endpoints: POST /predict | GET /health")
    app.run(host="0.0.0.0", port=5000, debug=False)
