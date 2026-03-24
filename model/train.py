"""
train.py — Full Two-Phase Training Script

Phase 1 (10 epochs): Only the custom classification head is trained.
              EfficientNetB4 base is frozen.

Phase 2 (5 epochs):  Top 30 layers of EfficientNet are unfrozen.
              Fine-tune end-to-end at a very low learning rate (1e-5).

Callbacks used in both phases:
  - EarlyStopping   (monitor val_auc, patience=3)
  - ReduceLROnPlateau (monitor val_loss, patience=2)
  - ModelCheckpoint   (saves best weights to deepfake_efficientnet_best.h5)

After training:
  - Final model saved as deepfake_efficientnet.h5
  - Training curves (accuracy + loss) saved as training_history.png
  - Test set evaluation: accuracy, AUC, precision, recall printed to console
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for Windows
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
import tensorflow as tf

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import build_efficientnet_model, build_lightweight_cnn, unfreeze_top_layers
from utils.preprocess import load_image_rgb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_DIR   = "archive/real_vs_fake/real-vs-fake/train"
VALID_DIR   = "archive/real_vs_fake/real-vs-fake/valid"
TEST_DIR    = "archive/real_vs_fake/real-vs-fake/test"

IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 16          # reduced for CPU
LIMIT_TRAIN = 2000        # images per class (None = all)
LIMIT_VALID = 500
LIMIT_TEST  = 500

PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 5
VAL_SPLIT     = 0.2

MODEL_TYPE    = "efficientnet"   # "efficientnet" | "cnn"
BEST_WEIGHTS  = "deepfake_efficientnet_best.h5"
FINAL_MODEL   = "deepfake_efficientnet.h5"
HISTORY_IMAGE = "training_history.png"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_images_from_subfolders(base_folder, image_size=IMAGE_SIZE,
                                 limit_per_class=None):
    """
    Load real/fake images from the standard folder structure.

    Expected layout:
        base_folder/
            real/   ← label 0
            fake/   ← label 1
    """
    images, labels = [], []
    classes = {"real": 0, "fake": 1}

    for label_name, label in classes.items():
        folder_path = os.path.join(base_folder, label_name)
        if not os.path.isdir(folder_path):
            print(f"[WARN] Folder not found: {folder_path} — skipping.")
            continue

        count = 0
        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            filepath = os.path.join(folder_path, filename)
            try:
                img = load_image_rgb(filepath)
                import cv2
                img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
                img = img.astype(np.float32) / 255.0
                images.append(img)
                labels.append(label)
                count += 1
                if limit_per_class and count >= limit_per_class:
                    break
            except Exception as e:
                print(f"[SKIP] {filepath}: {e}")
                continue

        print(f"  Loaded {count} '{label_name}' images from {folder_path}")

    if not images:
        raise RuntimeError(f"No images loaded from {base_folder}. "
                           "Check your dataset path.")

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)


# ---------------------------------------------------------------------------
# Callbacks factory
# ---------------------------------------------------------------------------

def make_callbacks(best_weights_path: str, phase: int = 1) -> list:
    monitor_early = "val_auc"
    monitor_lr    = "val_loss"

    early_stop = EarlyStopping(
        monitor=monitor_early,
        patience=3,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor_lr,
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    )
    checkpoint = ModelCheckpoint(
        filepath=best_weights_path,
        monitor=monitor_early,
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    return [early_stop, reduce_lr, checkpoint]


# ---------------------------------------------------------------------------
# Plot training curves
# ---------------------------------------------------------------------------

def plot_history(histories: list, save_path: str = HISTORY_IMAGE):
    """
    Plot accuracy and loss curves for Phase 1 + Phase 2 concatenated.
    """
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc     += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        loss    += h.history["loss"]
        val_loss+= h.history["val_loss"]

    epochs = range(1, len(acc) + 1)
    phase2_start = len(histories[0].history["accuracy"]) if len(histories) > 1 else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Deepfake Detection — Training History", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(epochs, acc,     "b-o", markersize=4, label="Train Accuracy")
    axes[0].plot(epochs, val_acc, "r-o", markersize=4, label="Val Accuracy")
    if phase2_start:
        axes[0].axvline(x=phase2_start + 0.5, color="grey", linestyle="--",
                        label="Fine-tune start")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(epochs, loss,     "b-o", markersize=4, label="Train Loss")
    axes[1].plot(epochs, val_loss, "r-o", markersize=4, label="Val Loss")
    if phase2_start:
        axes[1].axvline(x=phase2_start + 0.5, color="grey", linestyle="--",
                        label="Fine-tune start")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"\n[✓] Training curves saved → {save_path}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train():
    print("=" * 60)
    print(" Deepfake Detection — Training Script")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading training data …")
    X_train_all, y_train_all = load_images_from_subfolders(
        TRAIN_DIR, IMAGE_SIZE, LIMIT_TRAIN)

    print("[1/5] Loading validation data …")
    X_valid_all, y_valid_all = load_images_from_subfolders(
        VALID_DIR, IMAGE_SIZE, LIMIT_VALID)

    print("[1/5] Loading test data …")
    X_test, y_test = load_images_from_subfolders(
        TEST_DIR, IMAGE_SIZE, LIMIT_TEST)

    # Combine train + valid, then re-split
    X_combined = np.concatenate([X_train_all, X_valid_all])
    y_combined = np.concatenate([y_train_all, y_valid_all])
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, y_combined, test_size=VAL_SPLIT, random_state=42,
        stratify=y_combined
    )

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Build model ────────────────────────────────────────────────────────
    print(f"\n[2/5] Building {MODEL_TYPE} model …")
    if MODEL_TYPE == "efficientnet":
        model = build_efficientnet_model(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            freeze_base=True,
        )
    else:
        model = build_lightweight_cnn(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        )
    model.summary(line_length=90)

    # ── Phase 1 ───────────────────────────────────────────────────────────
    print(f"\n[3/5] Phase 1 — Training classification head ({PHASE1_EPOCHS} epochs) …")
    callbacks_p1 = make_callbacks(BEST_WEIGHTS, phase=1)
    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=PHASE1_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_p1,
        verbose=1,
    )

    # ── Phase 2 (EfficientNet only) ───────────────────────────────────────
    histories = [history1]
    if MODEL_TYPE == "efficientnet":
        print(f"\n[4/5] Phase 2 — Fine-tuning top-30 layers ({PHASE2_EPOCHS} epochs) …")
        model = unfreeze_top_layers(model, num_layers=30)
        callbacks_p2 = make_callbacks(BEST_WEIGHTS, phase=2)
        history2 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=PHASE2_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks_p2,
            verbose=1,
        )
        histories.append(history2)
    else:
        print("\n[4/5] Skipping Phase 2 (lightweight CNN — no base to fine-tune).")

    # ── Evaluation ────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating on test set …")
    results = model.evaluate(X_test, y_test, verbose=1)
    metric_names = model.metrics_names
    print("\n── Test Results ──────────────────────────")
    for name, value in zip(metric_names, results):
        print(f"  {name:12s}: {value:.4f}")
    print("──────────────────────────────────────────\n")

    # ── Save model ────────────────────────────────────────────────────────
    model.save(FINAL_MODEL)
    print(f"[✓] Final model saved → {FINAL_MODEL}")

    # ── Plot curves ───────────────────────────────────────────────────────
    plot_history(histories, HISTORY_IMAGE)

    print("\n[✓] Training complete.")
    return model


if __name__ == "__main__":
    # Reduce TensorFlow verbosity
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.get_logger().setLevel("ERROR")
    train()
