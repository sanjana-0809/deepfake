"""
helpers.py — Threshold Logic, Detection Hints & Result Formatting

Combines CNN score and FFT score into a final ensemble score,
applies a configurable threshold, and returns a clean result dictionary.
"""

from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Ensemble scoring
# ---------------------------------------------------------------------------

CNN_WEIGHT = 0.70
FFT_WEIGHT = 0.30


def compute_final_score(cnn_score: float, fft_score: float) -> float:
    """
    Weighted ensemble of CNN and FFT scores.

        final_score = 0.7 * cnn_score + 0.3 * fft_score

    Both inputs must be in [0, 1].  Returns a float in [0, 1].
    """
    cnn_score = float(max(0.0, min(1.0, cnn_score)))
    fft_score = float(max(0.0, min(1.0, fft_score)))
    return round(CNN_WEIGHT * cnn_score + FFT_WEIGHT * fft_score, 6)


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def classify(final_score: float, threshold: float = 0.5) -> str:
    """
    Returns "FAKE" if final_score >= threshold, else "REAL".
    """
    return "FAKE" if final_score >= threshold else "REAL"


def get_detection_hint(final_score: float) -> str:
    """
    Human-readable explanation of what the score likely means.

    Score ranges:
        > 0.85        → High confidence face swap / GAN artifact
        0.65 – 0.85   → Likely AI-generated (diffusion / GAN)
        0.50 – 0.65   → Borderline, low-confidence manipulation
        < 0.50        → Natural image, no manipulation detected
    """
    if final_score > 0.85:
        return "High confidence — Face swap or GAN artifact detected"
    elif final_score >= 0.65:
        return "Likely AI-generated — Diffusion model pattern found"
    elif final_score >= 0.50:
        return "Borderline — Possible manipulation, low confidence"
    else:
        return "Natural image — No manipulation detected"


# ---------------------------------------------------------------------------
# Result dictionary
# ---------------------------------------------------------------------------

def build_result(
    cnn_score: float,
    fft_score: float,
    threshold: float = 0.5,
    image_name: Optional[str] = None,
    face_found: bool = True,
) -> dict:
    """
    Compute the ensemble score and return a fully populated result dict.

    Args:
        cnn_score   : Model sigmoid output in [0, 1].
        fft_score   : FFT artifact score in [0, 1].
        threshold   : Decision boundary (default 0.5).
        image_name  : Optional filename string.
        face_found  : Whether a face was detected during preprocessing.

    Returns:
        {
            "label"             : "FAKE" | "REAL",
            "confidence_percent": str,     e.g. "94.7%"
            "final_score"       : float,   e.g. 0.947
            "cnn_score"         : float,
            "fft_score"         : float,
            "detection_hint"    : str,
            "threshold_used"    : float,
            "face_found"        : bool,
            "image_name"        : str | None,
            "timestamp"         : str,     ISO 8601
        }
    """
    final_score = compute_final_score(cnn_score, fft_score)
    label = classify(final_score, threshold)
    hint = get_detection_hint(final_score)

    # Confidence: probability of the predicted class
    if label == "FAKE":
        confidence = final_score
    else:
        confidence = 1.0 - final_score

    return {
        "label": label,
        "confidence_percent": f"{confidence * 100:.1f}%",
        "final_score": round(final_score, 4),
        "cnn_score": round(float(cnn_score), 4),
        "fft_score": round(float(fft_score), 4),
        "detection_hint": hint,
        "threshold_used": round(float(threshold), 2),
        "face_found": face_found,
        "image_name": image_name,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def build_batch_results(
    cnn_scores: list,
    fft_scores: list,
    threshold: float = 0.5,
    image_names: Optional[list] = None,
    face_found_flags: Optional[list] = None,
) -> list:
    """
    Build a list of result dicts for a batch of images.

    Args:
        cnn_scores       : List of float CNN scores.
        fft_scores       : List of float FFT scores.
        threshold        : Shared decision threshold.
        image_names      : Optional list of filenames.
        face_found_flags : Optional list of bools.

    Returns:
        List of result dicts (one per image).
    """
    n = len(cnn_scores)
    names = image_names if image_names else [None] * n
    faces = face_found_flags if face_found_flags else [True] * n

    return [
        build_result(
            cnn_score=cnn_scores[i],
            fft_score=fft_scores[i],
            threshold=threshold,
            image_name=names[i],
            face_found=faces[i],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def result_to_display_dict(result: dict) -> dict:
    """
    Slim version of the result dict suitable for Streamlit table display.
    """
    return {
        "Image": result.get("image_name") or "—",
        "Label": result["label"],
        "Confidence": result["confidence_percent"],
        "Detection Hint": result["detection_hint"],
        "CNN Score": result["cnn_score"],
        "FFT Score": result["fft_score"],
        "Threshold": result["threshold_used"],
        "Time": result["timestamp"],
    }
