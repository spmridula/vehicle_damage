import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)   # MobileNetV2 expects 224x224
BATCH_SIZE  = 32
EPOCHS      = 20           # EarlyStopping will stop this earlier if needed
NUM_CLASSES = 3
CLASS_NAMES = ["minor", "moderate", "severe"]
DATA_DIR    = "data"
MODEL_PATH  = "vehicle_damage_model.keras"

def validate_dataset():
    print("=== Validating dataset structure ===")
    if not os.path.exists(DATA_DIR):
        print(f"\n❌  Folder '{DATA_DIR}' not found in: {os.getcwd()}")
        print("    Create: data/train/minor, data/train/moderate, data/train/severe")
        print("    Create: data/val/minor, data/val/moderate, data/val/severe")
        sys.exit(1)
    errors = []
    for split in ["train", "val"]:
        for cls in CLASS_NAMES:
            cls_path = os.path.join(DATA_DIR, split, cls)
            if not os.path.exists(cls_path):
                errors.append(f"Missing: {cls_path}")
                continue
            imgs = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg",".jpeg",".png"))]
            if len(imgs) == 0:
                errors.append(f"No images in: {cls_path}")
            else:
                print(f"  ✅  {cls_path} → {len(imgs)} images")
    if errors:
        print("\n❌  Errors:"); [print(f"    • {e}") for e in errors]; sys.exit(1)
    print("✅  Dataset OK\n")


# ── STEP 1: DATA LOADING WITH AUGMENTATION ───────────────────────────────────
def build_data_generators():
    """
    ImageDataGenerator does two things:
    1. Rescales pixel values from 0-255 to 0-1 (required for neural networks)
    2. Augments training images: flips, zooms, rotations — makes the model
       more robust to real-world variation in how damage is photographed
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,       # flip left-right (damage can be on either side)
        zoom_range=0.2,             # slight zoom variation
        rotation_range=15,          # slight rotation (camera angle variation)
        brightness_range=[0.8, 1.2] # lighting variation
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)  # NO augmentation for validation

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",   # one-hot encoding for 3 classes
        classes=CLASS_NAMES
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES
    )

    return train_gen, val_gen


# ── STEP 2: BUILD MODEL WITH TRANSFER LEARNING ───────────────────────────────
def build_model():
    """
    Transfer learning strategy:
    - Load MobileNetV2 pre-trained on ImageNet (1.2M images, 1000 classes)
    - Freeze all its layers — we keep the learned features (edges, textures, shapes)
    - Add our own classification head on top for the 3 damage severity classes
    - This works because ImageNet features (detecting shapes, textures, objects)
      transfer well to car damage detection

    Why MobileNetV2 specifically?
    - Lightweight: fast training even on CPU
    - Good accuracy: better than training from scratch on small datasets
    - Industry-standard: widely used in production mobile/edge deployments
    """
    # Load base model without the top classification layer
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,           # remove ImageNet's 1000-class output layer
        weights="imagenet"           # use pre-trained weights
    )

    # Freeze base model — don't update these weights during initial training
    base_model.trainable = False

    # Build the full model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # reduces spatial dimensions to a vector
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),              # prevents overfitting
        layers.Dense(NUM_CLASSES, activation="softmax")  # 3-class probability output
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(model.summary())
    return model, base_model


# ── STEP 3: FINE-TUNING (after initial training) ─────────────────────────────
def fine_tune_model(model, base_model, train_gen, val_gen):
    """
    Fine-tuning: unfreeze the last 30 layers of the base model and
    train them at a very low learning rate. This lets the model adapt
    ImageNet features specifically to car damage patterns.
    Only do this AFTER the classification head has converged.
    """
    print("\n--- Fine-tuning: unfreezing last 30 layers ---")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Much lower learning rate to avoid destroying the pre-trained weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_ft = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
    )
    return history_ft


# ── STEP 4: TRAINING ─────────────────────────────────────────────────────────
def train(train_gen, val_gen, model):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy")
    ]

    print("\n--- Phase 1: Training classification head ---")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    return history


# ── STEP 5: PLOT TRAINING CURVES ─────────────────────────────────────────────
def plot_history(history, save_path="training_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"],     label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Model Accuracy over Epochs", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["loss"],     label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss over Epochs", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    validate_dataset()
    print("=== Vehicle Damage Severity Classifier ===")
    print(f"TensorFlow version: {tf.__version__}")

    train_gen, val_gen = build_data_generators()
    model, base_model = build_model()

    history = train(train_gen, val_gen, model)
    plot_history(history)

    # Optional fine-tuning — uncomment when initial training is complete
    # fine_tune_model(model, base_model, train_gen, val_gen)

    print(f"\nModel saved to: {MODEL_PATH}")
    print("Run gradcam.py next to generate heatmap explanations.")
