import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    full_train = datasets.ImageFolder(data_root / "Training", transform=train_tf)
    test_ds = datasets.ImageFolder(data_root / "Testing", transform=test_tf)

    print("Detected classes:", full_train.classes)

    train_size = int(0.80 * len(full_train))
    val_size = len(full_train) - train_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(0.30),
        nn.Linear(in_features, len(args.classes))
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience = 5
    bad_epochs = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        y_true = []
        y_pred = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = accuracy_score(y_true, y_pred)

        model.eval()
        total_loss = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                total_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_loss = total_loss / len(val_loader.dataset)
        val_acc = accuracy_score(y_true, y_pred)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), output_root / "best_model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(output_root / "best_model.pt", map_location=device))
    model.eval()

    all_y = []
    all_pred = []
    all_prob = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)

            all_y.extend(y.numpy())
            all_pred.extend(pred.cpu().numpy())
            all_prob.extend(prob.cpu().numpy())

    all_y = np.array(all_y)
    all_pred = np.array(all_pred)
    all_prob = np.array(all_prob)

    acc = accuracy_score(all_y, all_pred)
    weighted_f1 = f1_score(all_y, all_pred, average="weighted")

    try:
        y_bin = label_binarize(all_y, classes=list(range(len(args.classes))))
        macro_auc = roc_auc_score(y_bin, all_prob, average="macro", multi_class="ovr")
    except Exception:
        macro_auc = None

    metrics = {
        "accuracy": float(acc),
        "accuracy_percent": float(acc * 100),
        "weighted_f1": float(weighted_f1),
        "weighted_f1_percent": float(weighted_f1 * 100),
        "macro_auc": None if macro_auc is None else float(macro_auc),
        "classes": args.classes,
    }

    with open(output_root / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(all_y, all_pred, target_names=args.classes)
    with open(output_root / "classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(all_y, all_pred)
    pd.DataFrame(cm, index=args.classes, columns=args.classes).to_csv(output_root / "confusion_matrix.csv")

    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(len(args.classes)), args.classes, rotation=45, ha="right")
    plt.yticks(range(len(args.classes)), args.classes)
    plt.colorbar()
    for i in range(len(args.classes)):
        for j in range(len(args.classes)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_root / "confusion_matrix.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "accuracy_curve.png", dpi=300)
    plt.close()

    print("\nTest metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved results to: {output_root}")

if __name__ == "__main__":
    main()
