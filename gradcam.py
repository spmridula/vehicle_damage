# gradcam.py
# Generate Grad-CAM heatmaps to explain WHERE in the car the model looked
# when making a severity prediction.
#
# Grad-CAM = Gradient-weighted Class Activation Mapping
# It shows which pixels most influenced the prediction by tracking
# how gradients flow back through the last convolutional layer.
#
# Output: side-by-side image → original photo | heatmap overlay
# This is the "explainability layer" — critical for insurance use cases
# because you can show an adjuster exactly which damaged region drove the verdict.

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")

IMG_SIZE    = (224, 224)
CLASS_NAMES = ["Minor Damage", "Moderate Damage", "Severe Damage"]
MODEL_PATH  = "vehicle_damage_model.h5"


# ── CORE GRAD-CAM FUNCTION ────────────────────────────────────────────────────
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    How Grad-CAM works:
    1. Pass image through model, intercept output of the last conv layer
    2. Compute gradients of the predicted class score w.r.t. that layer's output
    3. Pool the gradients spatially (global average pooling over each feature map)
    4. Weight each feature map by its pooled gradient
    5. Apply ReLU — we only care about features that increase the class score
    6. The result is a heatmap showing which spatial regions drove the prediction

    Why the LAST convolutional layer?
    It has the highest-level semantic features (it "knows about" whole objects)
    while still retaining spatial information (earlier layers have better spatial
    resolution but lower-level features like edges, not semantic regions).
    """
    # Create a model that outputs:
    # - the last conv layer's output (feature maps)
    # - the final classification output
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradients of predicted class w.r.t. last conv layer output
    grads = tape.gradient(class_channel, last_conv_output)

    # Pool gradients over spatial dimensions — one scalar per feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by their importance score
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize to [0, 1] and apply ReLU
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(img_path, heatmap, alpha=0.4):
    """
    Superimpose the Grad-CAM heatmap onto the original image.
    Uses a jet colormap: blue=low activation, red=high activation.
    """
    # Load original image at display size
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)

    # Resize heatmap to match image size
    heatmap_resized = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    # Blend: original image + heatmap overlay
    superimposed = jet_heatmap * alpha + img_array * (1 - alpha)
    superimposed = tf.keras.utils.array_to_img(superimposed)
    return img_array, superimposed


def predict_and_explain(img_path, model, last_conv_layer_name="Conv_1", save_dir="gradcam_outputs"):
    """
    Full pipeline:
    1. Preprocess image
    2. Predict severity class + confidence scores
    3. Generate Grad-CAM heatmap
    4. Save side-by-side: original | heatmap overlay
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load and preprocess image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array_norm = np.expand_dims(img_array / 255.0, axis=0)

    # Predict
    preds = model.predict(img_array_norm, verbose=0)
    pred_class_idx = np.argmax(preds[0])
    pred_class_name = CLASS_NAMES[pred_class_idx]
    confidence = preds[0][pred_class_idx] * 100

    print(f"\nImage: {os.path.basename(img_path)}")
    print(f"Prediction: {pred_class_name} ({confidence:.1f}% confidence)")
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, preds[0])):
        bar = "█" * int(prob * 20)
        print(f"  {name:20s}: {prob*100:5.1f}%  {bar}")

    # Generate Grad-CAM
    heatmap = make_gradcam_heatmap(img_array_norm, model, last_conv_layer_name, pred_class_idx)
    original, overlaid = overlay_heatmap(img_path, heatmap)

    # Save side-by-side visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original.astype("uint8"))
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(overlaid)
    axes[1].set_title("Grad-CAM Heatmap\n(Red = high influence)", fontweight="bold", color="darkred")
    axes[1].axis("off")

    # Bar chart of probabilities
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = axes[2].barh(CLASS_NAMES, preds[0] * 100, color=colors)
    axes[2].set_xlabel("Confidence (%)")
    axes[2].set_title(f"Prediction: {pred_class_name}\n({confidence:.1f}% confident)", fontweight="bold")
    axes[2].set_xlim(0, 100)
    for bar, val in zip(bars, preds[0] * 100):
        axes[2].text(val + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=10)

    plt.suptitle(
        f"Vehicle Damage Assessment — {pred_class_name}",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    save_name = os.path.splitext(os.path.basename(img_path))[0] + "_gradcam.png"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM saved to: {save_path}")

    return pred_class_name, confidence, save_path


def find_last_conv_layer(model):
    """
    Automatically find the name of the last convolutional layer.
    For MobileNetV2 this is typically 'Conv_1' but may vary.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            # It's the base model — find last conv inside it
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return layer.name + "/" + sublayer.name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return "Conv_1"  # MobileNetV2 default


if __name__ == "__main__":
    import sys

    model = load_model(MODEL_PATH)
    print(f"Model loaded from: {MODEL_PATH}")

    # Find the last conv layer name automatically
    # For MobileNetV2 the base model is the first layer
    base_model = model.layers[0]
    last_conv = "Conv_1"  # Standard MobileNetV2 last conv layer name
    print(f"Using conv layer: {last_conv}")

    # If an image path is passed as argument, explain that specific image
    if len(sys.argv) > 1:
        img_paths = sys.argv[1:]
    else:
        # Default: look for any images in current directory
        img_paths = [f for f in os.listdir(".") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not img_paths:
            print("No images found. Pass image paths as arguments:")
            print("  python gradcam.py car1.jpg car2.jpg")
            exit()

    for img_path in img_paths:
        if os.path.exists(img_path):
            predict_and_explain(img_path, model, last_conv_layer_name=last_conv)
        else:
            print(f"File not found: {img_path}")
