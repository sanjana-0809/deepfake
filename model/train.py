"""
train.py — Full Two-Phase Training Script (Memory-Safe Version)
Uses ImageDataGenerator to load images in batches — no RAM overflow.

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
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import build_efficientnet_model, build_lightweight_cnn, unfreeze_top_layers

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_DIR     = "archive/real_vs_fake/real-vs-fake/train"
VALID_DIR     = "archive/real_vs_fake/real-vs-fake/valid"
TEST_DIR      = "archive/real_vs_fake/real-vs-fake/test"

IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 8           # tuned for 8 GB RAM with limited free memory

PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 5

MODEL_TYPE    = "cnn"       # "efficientnet" | "cnn"
BEST_WEIGHTS  = "deepfake_efficientnet_best.h5"
FINAL_MODEL   = "deepfake_efficientnet.h5"
HISTORY_IMAGE = "training_history.png"


# ---------------------------------------------------------------------------
# Data generators — loads images from disk in batches, no RAM overflow
# ---------------------------------------------------------------------------

def make_generators():
    """
    Returns train, validation, and test generators.
    Images are loaded batch-by-batch from disk — never all at once.
    """

    # Training generator with augmentation to improve generalisation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05,
    )

    # Validation and test — only rescale, no augmentation
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",        # real=0, fake=1
        shuffle=True,
        seed=42,
    )

    valid_gen = eval_datagen.flow_from_directory(
        VALID_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    return train_gen, valid_gen, test_gen


# ---------------------------------------------------------------------------
# Callbacks factory
# ---------------------------------------------------------------------------

def make_callbacks(best_weights_path: str) -> list:
    early_stop = EarlyStopping(
        monitor="val_auc",
        patience=3,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    )
    checkpoint = ModelCheckpoint(
        filepath=best_weights_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    return [early_stop, reduce_lr, checkpoint]


# ---------------------------------------------------------------------------
# Plot training curves
# ---------------------------------------------------------------------------

def plot_history(histories: list, save_path: str = HISTORY_IMAGE):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc      += h.history["accuracy"]
        val_acc  += h.history["val_accuracy"]
        loss     += h.history["loss"]
        val_loss += h.history["val_loss"]

    epochs = range(1, len(acc) + 1)
    phase2_start = len(histories[0].history["accuracy"]) if len(histories) > 1 else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Deepfake Detection — Training History", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, acc,     "b-o", markersize=4, label="Train Accuracy")
    axes[0].plot(epochs, val_acc, "r-o", markersize=4, label="Val Accuracy")
    if phase2_start:
        axes[0].axvline(x=phase2_start + 0.5, color="grey", linestyle="--", label="Fine-tune start")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, loss,     "b-o", markersize=4, label="Train Loss")
    axes[1].plot(epochs, val_loss, "r-o", markersize=4, label="Val Loss")
    if phase2_start:
        axes[1].axvline(x=phase2_start + 0.5, color="grey", linestyle="--", label="Fine-tune start")
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

    # ── Create data generators (no RAM loading!) ──────────────────────────
    print("\n[1/5] Setting up data generators …")
    train_gen, valid_gen, test_gen = make_generators()

    print(f"  Train batches : {len(train_gen)}  ({train_gen.samples} images)")
    print(f"  Valid batches : {len(valid_gen)}  ({valid_gen.samples} images)")
    print(f"  Test  batches : {len(test_gen)}  ({test_gen.samples} images)")

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
    print(f"\n[3/5] Phase 1 — Training ({PHASE1_EPOCHS} epochs) …")
    history1 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=PHASE1_EPOCHS,
        callbacks=make_callbacks(BEST_WEIGHTS),
        verbose=1,
    )

    # ── Phase 2 (EfficientNet only) ───────────────────────────────────────
    histories = [history1]
    if MODEL_TYPE == "efficientnet":
        print(f"\n[4/5] Phase 2 — Fine-tuning top-30 layers ({PHASE2_EPOCHS} epochs) …")
        model = unfreeze_top_layers(model, num_layers=30)
        history2 = model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=PHASE2_EPOCHS,
            callbacks=make_callbacks(BEST_WEIGHTS),
            verbose=1,
        )
        histories.append(history2)
    else:
        print("\n[4/5] Skipping Phase 2 (lightweight CNN — no base to fine-tune).")

    # ── Evaluation ────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating on test set …")
    results = model.evaluate(test_gen, verbose=1)
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
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.get_logger().setLevel("ERROR")
    train()
