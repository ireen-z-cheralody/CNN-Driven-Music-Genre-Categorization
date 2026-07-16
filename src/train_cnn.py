import numpy as np
import torch
import torch.nn as nn
import json
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from config import GENRES
from dataset import SpecDataset
from model import GenreCNN, count_parameters

DATA_PATH = "results/dataset_split.npz"
RESULTS_DIR = "results"
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3


def main():
    data = np.load(DATA_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    train_loader = DataLoader(SpecDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SpecDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(SpecDataset(X_test, y_test), batch_size=BATCH_SIZE)

    model = GenreCNN(n_classes=len(GENRES))
    print(f"Model has {count_parameters(model):,} trainable parameters")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(EPOCHS):
        # ---- training ----
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(xb)
        train_loss = total_loss / total
        train_acc = correct / total

        # ---- validation ----
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                loss = loss_fn(out, yb)
                val_loss += loss.item() * len(xb)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total += len(xb)
        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.3f} train_acc={train_acc:.3f} "
              f"| val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

    # ---- final test evaluation ----
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            all_preds.extend(out.argmax(1).tolist())
            all_labels.extend(yb.tolist())

    test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    report = classification_report(all_labels, all_preds, target_names=GENRES, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nFinal test accuracy: {test_acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=GENRES))

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- save quantitative results ----
    with open(os.path.join(RESULTS_DIR, "cnn_results.json"), "w") as f:
        json.dump({
            "n_parameters": count_parameters(model),
            "epochs": EPOCHS,
            "history": history,
            "test_accuracy": test_acc,
            "classification_report": report,
        }, f, indent=2)

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "cnn_model.pt"))

    # ---- learning curve plot (quantitative result #1) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "learning_curves.png"))
    print("Saved learning_curves.png")

    # ---- confusion matrix plot (qualitative result) ----
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(GENRES)))
    ax.set_yticks(range(len(GENRES)))
    ax.set_xticklabels(GENRES, rotation=45, ha="right")
    ax.set_yticklabels(GENRES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("CNN Confusion Matrix (Test Set)")
    for i in range(len(GENRES)):
        for j in range(len(GENRES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    print("Saved confusion_matrix.png")


if __name__ == "__main__":
    main()