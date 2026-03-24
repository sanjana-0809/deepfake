"""
preprocess.py — Face Detection & Image Preprocessing Pipeline

Strategy:
  1. OpenCV Haar Cascade (fast, CPU-friendly) — primary detector.
  2. MTCNN (deep-learning face detector) — fallback if OpenCV finds nothing.
  3. If neither finds a face → analyze the full image (no rejection).

Supports:
  - Single image processing.
  - Batch processing (list of images).
  - Any image content: human faces, animals, AI art, full-body, scenery.

All images are returned as float32 numpy arrays in [0, 1],
resized to TARGET_SIZE (default 224×224).
"""

import os
import cv2
import numpy as np
from typing import Union, List, Tuple

# MTCNN is optional; if import fails we silently skip it
try:
    from mtcnn import MTCNN
    _MTCNN_AVAILABLE = True
except ImportError:
    _MTCNN_AVAILABLE = False

TARGET_SIZE = (224, 224)
PADDING_RATIO = 0.20   # 20% padding around the detected face crop

# Path to OpenCV's bundled Haar Cascade XML
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ---------------------------------------------------------------------------
# Face detector initialisation (lazy singletons)
# ---------------------------------------------------------------------------

_haar_cascade: cv2.CascadeClassifier = None
_mtcnn_detector = None


def _get_haar_cascade() -> cv2.CascadeClassifier:
    global _haar_cascade
    if _haar_cascade is None:
        _haar_cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        if _haar_cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar Cascade from {_CASCADE_PATH}. "
                "Ensure opencv-python is properly installed."
            )
    return _haar_cascade


def _get_mtcnn() -> "MTCNN":
    global _mtcnn_detector
    if _mtcnn_detector is None and _MTCNN_AVAILABLE:
        _mtcnn_detector = MTCNN()
    return _mtcnn_detector


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _detect_face_opencv(img_bgr: np.ndarray) -> Union[Tuple[int, int, int, int], None]:
    """
    Detect the largest face using OpenCV Haar Cascade.

    Returns (x, y, w, h) of the largest detected face, or None.
    """
    cascade = _get_haar_cascade()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return None

    # Pick the largest detected face by area
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return tuple(faces[0])  # (x, y, w, h)


def _detect_face_mtcnn(img_rgb: np.ndarray) -> Union[Tuple[int, int, int, int], None]:
    """
    Detect the largest face using MTCNN.

    Returns (x, y, w, h) or None.
    """
    detector = _get_mtcnn()
    if detector is None:
        return None

    results = detector.detect_faces(img_rgb)
    if not results:
        return None

    # Pick the face with highest confidence
    results = sorted(results, key=lambda r: r["confidence"], reverse=True)
    box = results[0]["box"]  # [x, y, w, h]
    return tuple(box)        # (x, y, w, h)


# ---------------------------------------------------------------------------
# Crop + resize
# ---------------------------------------------------------------------------

def _crop_face_with_padding(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    padding_ratio: float = PADDING_RATIO,
) -> np.ndarray:
    """
    Crop the face from `img` with proportional padding.

    Args:
        img           : Original uint8 RGB image (H, W, 3).
        box           : (x, y, w, h) bounding box.
        padding_ratio : Fraction of face size to add as border.

    Returns:
        cropped       : uint8 RGB crop.
    """
    h_img, w_img = img.shape[:2]
    x, y, w, h = box

    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)

    return img[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Public API — single image
# ---------------------------------------------------------------------------

def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = TARGET_SIZE,
    return_face_found: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, bool]]:
    """
    Full preprocessing pipeline for a single image.

    Pipeline:
        detect face (OpenCV → MTCNN → full image)
        → crop with padding
        → resize to target_size
        → normalize to [0, 1]
        → add batch dimension → shape (1, H, W, 3)

    Args:
        image            : uint8 RGB image (H, W, 3).
        target_size      : (width, height) tuple, default (224, 224).
        return_face_found: If True, also return a bool indicating
                           whether a face was detected.

    Returns:
        preprocessed    : float32 array of shape (1, H, W, C).
        face_found      : bool  [only if return_face_found=True]
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    img_rgb = image.copy()
    if img_rgb.dtype != np.uint8:
        img_rgb = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # --- Step 1: Face detection ---
    face_found = True
    box = _detect_face_opencv(img_bgr)

    if box is None:
        # Fallback to MTCNN
        box = _detect_face_mtcnn(img_rgb)

    if box is None:
        # No face detected — use the full image
        face_found = False
        crop = img_rgb
    else:
        crop = _crop_face_with_padding(img_rgb, box)

    # --- Step 2: Resize ---
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

    # --- Step 3: Normalize ---
    normalized = resized.astype(np.float32) / 255.0

    # --- Step 4: Add batch dim ---
    batched = np.expand_dims(normalized, axis=0)  # (1, H, W, 3)

    if return_face_found:
        return batched, face_found
    return batched


def load_and_preprocess_from_path(
    image_path: str,
    target_size: Tuple[int, int] = TARGET_SIZE,
    return_face_found: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, bool]]:
    """
    Load an image from disk and run the full preprocessing pipeline.

    Args:
        image_path       : Absolute or relative path to the image file.
        target_size      : Resize target (W, H).
        return_face_found: Also return whether a face was found.

    Returns:
        preprocessed (and optionally face_found).
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not decode image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return preprocess_image(img_rgb, target_size=target_size,
                            return_face_found=return_face_found)


# ---------------------------------------------------------------------------
# Public API — batch processing
# ---------------------------------------------------------------------------

def preprocess_batch(
    images: List[np.ndarray],
    target_size: Tuple[int, int] = TARGET_SIZE,
) -> Tuple[np.ndarray, List[bool]]:
    """
    Preprocess a list of uint8 RGB images.

    Args:
        images      : List of uint8 RGB arrays (H, W, 3).
        target_size : Resize target (W, H).

    Returns:
        batch_array : float32 array of shape (N, H, W, 3).
        face_flags  : List of bool indicating face detection per image.
    """
    processed = []
    face_flags = []

    for img in images:
        arr, found = preprocess_image(img, target_size=target_size,
                                      return_face_found=True)
        processed.append(arr[0])   # remove the leading batch dim
        face_flags.append(found)

    batch_array = np.stack(processed, axis=0)  # (N, H, W, 3)
    return batch_array, face_flags


def load_image_rgb(image_path: str) -> np.ndarray:
    """
    Utility: load any image from disk and return as uint8 RGB.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
