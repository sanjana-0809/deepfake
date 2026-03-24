"""
model.py — Deepfake Detection Model
Supports EfficientNetB4 (recommended) and a lightweight CNN fallback for CPU-only users.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.metrics import AUC, Precision, Recall


def build_efficientnet_model(input_shape=(224, 224, 3), freeze_base=True):
    """
    EfficientNetB4 backbone with a custom classification head.

    Phase 1 training: freeze_base=True  → only the head is trained.
    Phase 2 fine-tuning: call unfreeze_top_layers() afterwards.

    Architecture:
        EfficientNetB4 (pretrained ImageNet)
        → GlobalAveragePooling2D
        → Dense(512, relu) → BatchNorm → Dropout(0.4)
        → Dense(256, relu) → BatchNorm → Dropout(0.3)
        → Dense(1, sigmoid)
    """
    base_model = EfficientNetB4(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = not freeze_base

    inputs = tf.keras.Input(shape=input_shape, name="image_input")

    # EfficientNet expects pixel values in [0, 255] but we pass [0, 1];
    # rescale back so the built-in preprocessing works correctly.
    x = layers.Rescaling(255.0)(inputs)
    x = base_model(x, training=False)

    # --- Classification head ---
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="dense_512")(x)
    x = layers.BatchNormalization(name="bn_512")(x)
    x = layers.Dropout(0.4, name="dropout_04")(x)

    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="dense_256")(x)
    x = layers.BatchNormalization(name="bn_256")(x)
    x = layers.Dropout(0.3, name="dropout_03")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs, outputs, name="EfficientNetB4_Deepfake")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
        ],
    )
    return model


def unfreeze_top_layers(model, num_layers=30):
    """
    Unfreeze the top `num_layers` of the EfficientNetB4 base for fine-tuning.
    Call this before Phase 2 training and recompile with a lower lr.
    """
    base_model = model.get_layer("efficientnetb4")
    base_model.trainable = True

    # Freeze everything except the last num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
        ],
    )
    return model


def build_lightweight_cnn(input_shape=(224, 224, 3)):
    """
    Lightweight CNN fallback for CPU-only users with limited RAM.
    Deeper than the base project but still fast to train/infer on CPU.

    Architecture:
        Conv(32) → MaxPool
        Conv(64) → MaxPool
        Conv(128) → MaxPool
        Conv(256) → MaxPool
        GlobalAvgPool
        Dense(256) → BatchNorm → Dropout(0.4)
        Dense(128) → BatchNorm → Dropout(0.3)
        Dense(1, sigmoid)
    """
    model = models.Sequential(name="LightweightCNN_Deepfake")

    model.add(layers.Conv2D(32, (3, 3), activation="relu",
                            padding="same", input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(256, activation="relu",
                           kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(128, activation="relu",
                           kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
        ],
    )
    return model


def get_model(model_type: str = "efficientnet", freeze_base: bool = True):
    """
    Factory function. Returns the requested model.

    Args:
        model_type : "efficientnet" | "cnn"
        freeze_base: relevant only for efficientnet; True = Phase 1 (head only)
    """
    if model_type == "efficientnet":
        return build_efficientnet_model(freeze_base=freeze_base)
    elif model_type == "cnn":
        return build_lightweight_cnn()
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'efficientnet' or 'cnn'.")
