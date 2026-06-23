# Vehicle Damage Severity Classifier

An end-to-end computer vision project that classifies vehicle damage severity from photos using Transfer Learning (MobileNetV2), explains predictions using Grad-CAM heatmaps, and delivers results through a web interface — targeting automotive insurance claims workflows.

---

## What This Project Does

> Given a photo of a damaged vehicle, classify damage as Minor / Moderate / Severe, highlight which region of the car drove the prediction, and present results in a format an insurance adjuster can act on.

| Stage | Technique | Output |
|---|---|---|
| Feature extraction | MobileNetV2 pre-trained on ImageNet | 1280-dim feature vector per image |
| Classification | Fine-tuned dense layers | Minor / Moderate / Severe + probabilities |
| Explainability | Grad-CAM on last conv layer | Heatmap showing which damage region matters |
| Interface | Gradio web app | Upload photo → instant assessment |

---

## Files

```
vehicle_damage/
├── train_model.py   # Fine-tune MobileNetV2 on car damage dataset
├── gradcam.py       # Generate Grad-CAM heatmaps for any image
├── app.py           # Gradio web interface
└── README.md
```

---

## Dataset

**CarDD — Car Damage Detection** (Kaggle)
https://www.kaggle.com/datasets/anujms/car-damage-detection

Organize into this structure:
```
data/
├── train/
│   ├── minor/
│   ├── moderate/
│   └── severe/
└── val/
    ├── minor/
    ├── moderate/
    └── severe/
```

---

## How to Run

```bash
# Install dependencies
pip install tensorflow gradio matplotlib pillow numpy

# Train the model
python train_model.py

# Generate Grad-CAM for specific images
python gradcam.py car_photo.jpg

# Launch web interface
python app.py
```

---

## Key Technical Decisions

**Why MobileNetV2?** Lightweight, fast training, strong accuracy on visual tasks. Pre-trained on 1.2M ImageNet images — those features (edges, shapes, textures) transfer well to car damage.

**Why Transfer Learning?** A car damage dataset has thousands of images, not millions. Training from scratch would overfit. Freezing ImageNet weights and fine-tuning only the top layers gives much better results with far less data.

**Why Grad-CAM?** Insurance is a regulated industry. You cannot tell a claims officer "the AI said severe." You need to point to exactly which damaged panel drove the verdict. Grad-CAM does that — it shows you the heatmap, not just the label.

**Why Gradio?** A model sitting in a Jupyter notebook has zero business value. Gradio turns it into a usable tool in 20 lines of Python.
