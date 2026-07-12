import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import warnings
warnings.filterwarnings("ignore")

DATA_DIR    = "data"
MODEL_PATH  = "vehicle_damage_model.pth"
CLASS_NAMES = ["minor", "moderate", "severe"]
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_dataset():
    print("=== Validating dataset ===")
    if not os.path.exists(DATA_DIR):
        print(f"❌  '{DATA_DIR}' folder not found in {os.getcwd()}")
        print("    Create: data/train/minor, data/train/moderate, data/train/severe")
        print("            data/val/minor,   data/val/moderate,   data/val/severe")
        sys.exit(1)
    errors = []
    for split in ["train", "val"]:
        for cls in CLASS_NAMES:
            path = os.path.join(DATA_DIR, split, cls)
            if not os.path.exists(path):
                errors.append(f"Missing folder: {path}")
                continue
            imgs = [f for f in os.listdir(path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            if not imgs:
                errors.append(f"No images in: {path}")
            else:
                print(f"  ✅  {path:45s} → {len(imgs)} images")
    if errors:
        print("\n❌  Fix these before training:")
        for e in errors:
            print(f"    • {e}")
        sys.exit(1)
    print(f"✅  Dataset OK  |  Device: {DEVICE}\n")


def get_loaders():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_tf)
    print(f"Training samples  : {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Class mapping     : {train_ds.class_to_idx}\n")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader


def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 3)
    )
    model = model.to(DEVICE)
    print(f"Model built. Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    return model


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def train(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    best_val_acc = 0.0
    patience_counter = 0
    patience = 5
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("--- Phase 1: Training classifier head ---")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.3f}", end="")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("  ✅ saved")
            patience_counter = 0
        else:
            print()
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break
    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    return history


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_acc"], label="Train")
    ax1.plot(history["val_acc"],   label="Val")
    ax1.set_title("Accuracy", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(history["train_loss"], label="Train")
    ax2.plot(history["val_loss"],   label="Val")
    ax2.set_title("Loss", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()
    print("Training curves saved: training_curves.png")


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}\n")
    validate_dataset()
    train_loader, val_loader = get_loaders()
    model = build_model()
    history = train(model, train_loader, val_loader)
    plot_history(history)
    print(f"\n✅  Model saved to: {MODEL_PATH}")
    print("Next: python gradcam.py <your_car_image.jpg>")