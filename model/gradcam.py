"""
gradcam.py — Grad-CAM Explainability for the Deepfake Detection Model

Produces a color heatmap overlaid on the original image that highlights
which spatial regions influenced the FAKE / REAL decision.

Works for:
  - EfficientNetB4  (last conv layer: 'top_conv')
  - LightweightCNN  (last conv layer auto-detected)
"""

import numpy as np
import cv2
import tensorflow as tf


# ---------------------------------------------------------------------------
# Layer name resolution
# ---------------------------------------------------------------------------

def _get_last_conv_layer_name(model: tf.keras.Model) -> str:
    """Return the name of the last Conv2D layer in the model."""
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
        # EfficientNet wraps its layers inside a sub-model
        elif hasattr(layer, "layers"):
            for sub in layer.layers:
                if isinstance(sub, tf.keras.layers.Conv2D):
                    last_conv = sub.name
    if last_conv is None:
        raise ValueError("No Conv2D layer found in the model.")
    return last_conv


def _resolve_grad_model(model: tf.keras.Model) -> tuple:
    """
    Returns (grad_model, last_conv_layer_name) regardless of whether
    EfficientNet layers are nested inside a sub-model wrapper.
    """
    model_name = model.name.lower()

    if "efficientnet" in model_name:
        # EfficientNetB4 last conv layer is 'top_conv' inside the sub-model
        try:
            base = model.get_layer("efficientnetb4")
            last_conv_name = "top_conv"
            last_conv_layer = base.get_layer(last_conv_name)

            # Build a grad model that outputs (last conv activations, predictions)
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[last_conv_layer.output, model.output],
            )
        except Exception:
            # Fallback: walk through all layers
            last_conv_name = _get_last_conv_layer_name(model)
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(last_conv_name).output, model.output],
            )
    else:
        last_conv_name = _get_last_conv_layer_name(model)
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_name).output, model.output],
        )

    return grad_model, last_conv_name


# ---------------------------------------------------------------------------
# Core Grad-CAM computation
# ---------------------------------------------------------------------------

def compute_gradcam(model: tf.keras.Model, img_array: np.ndarray) -> np.ndarray:
    """
    Compute the raw Grad-CAM heatmap for a single preprocessed image.

    Args:
        model     : Loaded Keras model (EfficientNetB4 or LightweightCNN).
        img_array : Preprocessed image of shape (1, H, W, 3), values in [0, 1].

    Returns:
        heatmap   : 2-D float32 array in [0, 1], shape (h, w).
    """
    grad_model, _ = _resolve_grad_model(model)

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        # Binary classification: single sigmoid output
        loss = predictions[:, 0]

    # Gradients of the output w.r.t. the last conv feature maps
    grads = tape.gradient(loss, conv_outputs)          # (1, h, w, C)

    # Global average pooling of gradients over spatial dims
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    # Weight each channel by its pooled gradient
    conv_outputs = conv_outputs[0]                     # (h, w, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap = tf.squeeze(heatmap)                      # (h, w)

    # ReLU + normalise to [0, 1]
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap.numpy()

    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val

    return heatmap.astype(np.float32)


# ---------------------------------------------------------------------------
# Overlay generation
# ---------------------------------------------------------------------------

def overlay_heatmap_on_image(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Resize the heatmap to match `original_img` and blend them.

    Args:
        original_img : BGR or RGB uint8 image, shape (H, W, 3).
        heatmap      : 2-D float32 array in [0, 1].
        alpha        : Blend weight for the heatmap overlay.
        colormap     : OpenCV colormap constant.

    Returns:
        overlaid     : uint8 RGB image with heatmap overlay.
    """
    h, w = original_img.shape[:2]

    # Scale heatmap to [0, 255] and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)  # BGR

    # Ensure original is BGR for blending
    if original_img.shape[2] == 3:
        original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    else:
        original_bgr = original_img.copy()

    overlaid_bgr = cv2.addWeighted(original_bgr, 1 - alpha,
                                   heatmap_colored, alpha, 0)
    overlaid_rgb = cv2.cvtColor(overlaid_bgr, cv2.COLOR_BGR2RGB)
    return overlaid_rgb


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------

def generate_gradcam_overlay(
    model: tf.keras.Model,
    img_array: np.ndarray,
    original_img: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Full pipeline: preprocess → Grad-CAM → overlay.

    Args:
        model        : Loaded Keras model.
        img_array    : Preprocessed image (1, 224, 224, 3) in [0, 1].
        original_img : Original display image (H, W, 3) uint8 RGB.
        alpha        : Heatmap blend strength (0 = invisible, 1 = full heatmap).

    Returns:
        overlaid_rgb : uint8 RGB image (same size as original_img).
    """
    heatmap = compute_gradcam(model, img_array)
    overlaid = overlay_heatmap_on_image(original_img, heatmap, alpha=alpha)
    return overlaid
