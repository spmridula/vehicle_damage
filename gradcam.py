import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH  = "vehicle_damage_model.pth"
CLASS_NAMES = ["Minor Damage", "Moderate Damage", "Severe Damage"]
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def load_model():
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 3)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def preprocess(img_path):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    img = Image.open(img_path).convert("RGB")
    return tf(img).unsqueeze(0).to(DEVICE), img


def make_gradcam_heatmap(model, img_tensor, pred_class_idx):
    activations = {}
    gradients   = {}
    target_layer = model.features[18][0]

    def forward_hook(module, input, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    model.zero_grad()
    output[0, pred_class_idx].backward()
    fh.remove()
    bh.remove()

    pooled_grads = gradients["value"].mean(dim=[0, 2, 3])
    acts = activations["value"][0]
    for i, w in enumerate(pooled_grads):
        acts[i] *= w

    heatmap = acts.mean(dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def overlay_heatmap(pil_img, heatmap, alpha=0.4):
    img_array = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    heatmap_large = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
        dtype=np.float32
    ) / 255.0
    jet_rgb = (plt.colormaps["jet"](heatmap_large)[:, :, :3] * 255).astype(np.float32)
    blended = np.clip(jet_rgb * alpha + img_array * (1 - alpha), 0, 255).astype(np.uint8)
    return blended


def predict_and_explain(img_path, model, save_dir="gradcam_outputs"):
    os.makedirs(save_dir, exist_ok=True)
    img_tensor, pil_img = preprocess(img_path)

    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100

    print(f"\nImage   : {os.path.basename(img_path)}")
    print(f"Verdict : {pred_class}  ({confidence:.1f}%)")
    for name, prob in zip(CLASS_NAMES, probs):
        bar = "█" * int(prob * 20)
        print(f"  {name:20s}: {prob*100:5.1f}%  {bar}")

    heatmap  = make_gradcam_heatmap(model, img_tensor, pred_idx)
    overlaid = overlay_heatmap(pil_img, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    axes[0].set_title("Original Image"); axes[0].axis("off")
    axes[1].imshow(overlaid)
    axes[1].set_title("Grad-CAM  (red = model focus)"); axes[1].axis("off")
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = axes[2].barh(CLASS_NAMES, probs * 100, color=colors, height=0.5)
    axes[2].set_xlabel("Confidence (%)")
    axes[2].set_xlim(0, 105)
    axes[2].set_title(f"{pred_class} — {confidence:.1f}%", fontweight="bold")
    for bar, val in zip(bars, probs * 100):
        axes[2].text(val + 1, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=10)
    axes[2].spines[["top", "right"]].set_visible(False)
    plt.suptitle(f"Vehicle Damage — {pred_class}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_name = os.path.splitext(os.path.basename(img_path))[0] + "_gradcam.png"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved  : {save_path}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌  {MODEL_PATH} not found. Run train_model.py first.")
        sys.exit(1)
    model = load_model()
    print(f"✅  Model loaded from {MODEL_PATH}")
    img_paths = sys.argv[1:] if len(sys.argv) > 1 else \
        [f for f in os.listdir(".") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not img_paths:
        print("Usage: python gradcam.py car1.jpg car2.jpg")
        sys.exit(0)
    for p in img_paths:
        if os.path.exists(p):
            predict_and_explain(p, model)
        else:
            print(f"Not found: {p}")