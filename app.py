# app.py
# Gradio interface: upload a car photo → get damage severity + Grad-CAM heatmap
# This is what makes the model usable by insurance adjusters, not just data scientists.
#
# Run: python app.py
# Opens at: http://127.0.0.1:7860

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Gradio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gradio as gr
from PIL import Image
import io
import warnings
warnings.filterwarnings("ignore")

IMG_SIZE    = (224, 224)
CLASS_NAMES = ["Minor Damage", "Moderate Damage", "Severe Damage"]
MODEL_PATH  = "vehicle_damage_model.h5"

# Descriptions for each damage class — explains the verdict to an adjuster
CLASS_DESCRIPTIONS = {
    "Minor Damage": (
        "Surface-level damage only. Scratches, small dents, paint chips. "
        "Typically repaired without panel replacement. "
        "Estimated repair cost: ₹5,000 – ₹25,000"
    ),
    "Moderate Damage": (
        "Structural panel damage. Bumper cracks, broken lights, door deformation. "
        "Panel replacement likely needed. "
        "Estimated repair cost: ₹25,000 – ₹1,50,000"
    ),
    "Severe Damage": (
        "Major structural or mechanical damage. Frame damage, airbag deployment, "
        "engine compartment involved. May be a total loss. "
        "Estimated repair cost: ₹1,50,000+ or total loss declaration"
    )
}

SEVERITY_EMOJI = {
    "Minor Damage":    "🟢",
    "Moderate Damage": "🟡",
    "Severe Damage":   "🔴"
}

# Load model once at startup
print("Loading model...")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Could not load model: {e}")
    print("Train the model first: python train_model.py")
    model = None


def make_gradcam_heatmap(img_array_norm, model):
    """Generate Grad-CAM heatmap from normalized image array."""
    # Get the MobileNetV2 base model (first layer in Sequential)
    base_model = model.layers[0]

    # Build grad model intercepting the last conv layer of MobileNetV2
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[base_model.get_layer("Conv_1").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array_norm)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def analyze_damage(uploaded_image):
    """
    Main function called by Gradio when user uploads an image.
    Returns: verdict text, annotated image with heatmap, confidence chart
    """
    if model is None:
        return "❌ Model not loaded. Run train_model.py first.", None, None

    if uploaded_image is None:
        return "Please upload a car damage image.", None, None

    # Preprocess
    img = uploaded_image.resize(IMG_SIZE)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:  # handle RGBA images
        img_array = img_array[:, :, :3]
    img_array_norm = np.expand_dims(img_array / 255.0, axis=0).astype(np.float32)

    # Predict
    preds = model.predict(img_array_norm, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(preds[pred_idx]) * 100

    # Generate Grad-CAM
    try:
        heatmap, _ = make_gradcam_heatmap(img_array_norm, model)

        # Overlay heatmap on original image
        heatmap_resized = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap_resized]
        jet_heatmap_img = Image.fromarray(np.uint8(jet_heatmap * 255)).resize(IMG_SIZE)
        jet_array = np.array(jet_heatmap_img, dtype=np.float32)
        blended = jet_array * 0.4 + img_array * 0.6
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        overlay_img = Image.fromarray(blended)
    except Exception:
        overlay_img = img  # fallback: just show original if Grad-CAM fails

    # Build verdict text
    emoji = SEVERITY_EMOJI[pred_class]
    desc = CLASS_DESCRIPTIONS[pred_class]
    verdict = f"""
{emoji} DAMAGE ASSESSMENT VERDICT
{'━' * 40}
Severity Class  : {pred_class}
Confidence      : {confidence:.1f}%

Assessment      : {desc}

{'━' * 40}
Confidence Breakdown:
  🟢 Minor     : {preds[0]*100:.1f}%
  🟡 Moderate  : {preds[1]*100:.1f}%
  🔴 Severe    : {preds[2]*100:.1f}%
{'━' * 40}
⚠️  Model recommendation only. Human adjuster review required for final claim decision.
    """.strip()

    # Confidence bar chart
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax.barh(CLASS_NAMES, preds * 100, color=colors, edgecolor="white", height=0.5)
    ax.set_xlabel("Confidence (%)", fontsize=11)
    ax.set_xlim(0, 105)
    ax.set_title(f"Prediction: {pred_class}  ({confidence:.1f}% confident)",
                 fontweight="bold", fontsize=12)
    for bar, val in zip(bars, preds * 100):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    return verdict, overlay_img, fig


# ── GRADIO UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Vehicle Damage Severity Classifier", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🚗 Vehicle Damage Severity Classifier
    ### AI-powered damage assessment for insurance claims processing
    Upload a photo of a damaged vehicle. The model will:
    - Classify damage severity: **Minor / Moderate / Severe**
    - Show **Grad-CAM heatmap** highlighting which region drove the prediction
    - Provide a **confidence breakdown** across all three classes
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload Vehicle Photo")
            analyze_btn = gr.Button("🔍 Analyze Damage", variant="primary", size="lg")

        with gr.Column(scale=1):
            verdict_output = gr.Textbox(
                label="Damage Assessment Report",
                lines=18,
                show_copy_button=True
            )

    with gr.Row():
        heatmap_output = gr.Image(label="Grad-CAM Heatmap (Red = high influence region)")
        chart_output   = gr.Plot(label="Confidence Score Breakdown")

    analyze_btn.click(
        fn=analyze_damage,
        inputs=img_input,
        outputs=[verdict_output, heatmap_output, chart_output]
    )

    gr.Markdown("""
    ---
    **How to read the Grad-CAM heatmap:**
    Red/orange regions = areas the model focused on most when making its prediction.
    This lets an adjuster verify the model is looking at actual damage, not background.

    **Tech stack:** TensorFlow · MobileNetV2 (Transfer Learning) · Grad-CAM · Gradio
    """)

if __name__ == "__main__":
    demo.launch(share=False)
