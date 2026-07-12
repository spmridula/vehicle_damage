import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH  = "vehicle_damage_model.pth"
CLASS_NAMES = ["Minor Damage", "Moderate Damage", "Severe Damage"]
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

CLASS_DESCRIPTIONS = {
    "Minor Damage":    "Surface scratches, small dents, paint chips. Est: ₹5,000–₹25,000",
    "Moderate Damage": "Bumper cracks, broken lights, door deformation. Est: ₹25,000–₹1,50,000",
    "Severe Damage":   "Frame/airbag/engine damage. Possible total loss. Est: ₹1,50,000+"
}
EMOJI = {"Minor Damage": "🟢", "Moderate Damage": "🟡", "Severe Damage": "🔴"}

print("Loading model...")
try:
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, 3))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("✅  Model loaded.")
except Exception as e:
    print(f"❌  Could not load model: {e}\n    Run train_model.py first.")
    model = None

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


def make_gradcam(model, img_tensor, pred_idx):
    activations, gradients = {}, {}
    target_layer = model.features[18][0]
    fh = target_layer.register_forward_hook(
        lambda m, i, o: activations.update({"v": o.detach()}))
    bh = target_layer.register_backward_hook(
        lambda m, gi, go: gradients.update({"v": go[0].detach()}))
    out = model(img_tensor)
    model.zero_grad()
    out[0, pred_idx].backward()
    fh.remove(); bh.remove()
    pooled = gradients["v"].mean(dim=[0, 2, 3])
    acts   = activations["v"][0]
    for i, w in enumerate(pooled):
        acts[i] *= w
    heatmap = acts.mean(dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def analyze_damage(uploaded_image):
    if model is None:
        return "❌ Model not loaded. Run train_model.py first.", None, None
    if uploaded_image is None:
        return "Please upload a car damage image.", None, None

    pil_img    = uploaded_image.convert("RGB")
    img_tensor = tf(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100

    try:
        heatmap  = make_gradcam(model, img_tensor, pred_idx)
        img_arr  = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
        hm_large = np.array(
            Image.fromarray(np.uint8(255 * heatmap)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
            dtype=np.float32) / 255.0
        jet_rgb  = (cm.get_cmap("jet")(hm_large)[:, :, :3] * 255).astype(np.float32)
        overlay  = Image.fromarray(np.clip(jet_rgb * 0.4 + img_arr * 0.6, 0, 255).astype(np.uint8))
    except Exception:
        overlay = pil_img.resize((IMG_SIZE, IMG_SIZE))

    verdict = (
        f"{EMOJI[pred_class]} DAMAGE ASSESSMENT\n"
        f"{'━'*36}\n"
        f"Severity  : {pred_class}\n"
        f"Confidence: {confidence:.1f}%\n\n"
        f"{CLASS_DESCRIPTIONS[pred_class]}\n\n"
        f"{'━'*36}\n"
        f"  🟢 Minor    : {probs[0]*100:.1f}%\n"
        f"  🟡 Moderate : {probs[1]*100:.1f}%\n"
        f"  🔴 Severe   : {probs[2]*100:.1f}%\n"
        f"{'━'*36}\n"
        f"⚠️  Human review required."
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax.barh(CLASS_NAMES, probs * 100, color=colors, height=0.5)
    ax.set_xlabel("Confidence (%)"); ax.set_xlim(0, 105)
    ax.set_title(f"{pred_class} — {confidence:.1f}%", fontweight="bold")
    for bar, val in zip(bars, probs * 100):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    return verdict, overlay, fig


with gr.Blocks(title="Vehicle Damage Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚗 Vehicle Damage Severity Classifier\nUpload a damaged car photo → instant assessment + Grad-CAM heatmap.")
    with gr.Row():
        img_input = gr.Image(type="pil", label="Upload Vehicle Photo")
        btn       = gr.Button("🔍 Analyze", variant="primary", size="lg")
    verdict_out = gr.Textbox(label="Assessment Report", lines=16)
    with gr.Row():
        heatmap_out = gr.Image(label="Grad-CAM (red = model focus)")
        chart_out   = gr.Plot(label="Confidence Scores")
    btn.click(fn=analyze_damage, inputs=img_input,
              outputs=[verdict_out, heatmap_out, chart_out])

if __name__ == "__main__":
    demo.launch()