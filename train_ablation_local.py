#!/usr/bin/env python3
"""
Local ablation training script for M4 Mac Mini.
Runs classical-only models (no quantum layer needed).

Usage:
  python train_ablation_local.py --model classical_conv --epochs 9
  python train_ablation_local.py --model param_linear --epochs 9
  python train_ablation_local.py --model classical_conv --epochs 1 --test  # quick test
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from src.ablation_models import ClassicalBaselineNet, ParamMatchedLinearNet
from src import config


# ---- Label Smoothing (same as V7 training) ----

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_pred = torch.log_softmax(pred, dim=-1)
        nll_loss = -log_pred.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_pred.mean(dim=-1)
        return ((1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss).mean()


# ---- Mixup (same as V7 training) ----

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# ---- Data Loading ----

def load_data():
    """Load dataset using project config paths."""
    from src.dataset import load_images_from_folder

    train_img, train_lbl = load_images_from_folder(
        config.TRAIN_PATH, config.TAGS, config.IMAGE_SIZE
    )
    test_img, test_lbl = load_images_from_folder(
        config.TEST_PATH, config.TAGS, config.IMAGE_SIZE
    )

    train_img = torch.tensor(train_img / 255.0, dtype=torch.float32).unsqueeze(1)
    test_img = torch.tensor(test_img / 255.0, dtype=torch.float32).unsqueeze(1)
    train_lbl = torch.tensor(train_lbl, dtype=torch.long)
    test_lbl = torch.tensor(test_lbl, dtype=torch.long)

    train_ds = TensorDataset(train_img, train_lbl)
    test_ds = TensorDataset(test_img, test_lbl)

    val_size = int(config.VALIDATION_SPLIT * len(train_ds))
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])

    bs = config.BATCH_SIZE
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(val_ds, batch_size=bs, shuffle=False),
        DataLoader(test_ds, batch_size=bs, shuffle=False),
    )


# ---- Training ----

def train(model, train_loader, val_loader, test_loader, args):
    device = torch.device("cpu")  # M4 Mac, no CUDA
    model = model.to(device)

    # Same optimizer as V7 classical params
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    patience_counter = 0
    patience_limit = 15  # early stop if no improvement for 15 epochs

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}")
    print(f"Total params: {total_params}, Trainable: {trainable}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print("=" * 50)

    results = {
        "model": args.model,
        "total_params": total_params,
        "trainable_params": trainable,
        "epochs": [],
    }

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Mixup with 50% probability (same as V7)
            if np.random.random() < 0.5:
                images, y_a, y_b, lam = mixup_data(images, labels)
                output = model(images)
                loss = lam * criterion(output, y_a) + (1 - lam) * criterion(output, y_b)
            else:
                output = model(images)
                loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")

        scheduler.step()
        train_loss = total_loss / total
        train_acc = 100. * correct / total

        # ---- Validate ----
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/best_ablation_{args.model}.pth")
            print(f"  NEW BEST: {val_acc:.2f}%")
        else:
            patience_counter += 1

        results["epochs"].append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
        })

        if patience_counter >= patience_limit:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience_limit} epochs)")
            break

    # ---- Test ----
    print("\nFinal test evaluation...")
    model.load_state_dict(torch.load(f"models/best_ablation_{args.model}.pth", weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.2f}%")

    results["best_val_acc"] = round(best_val_acc, 2)
    results["test_acc"] = round(test_acc, 2)

    # Save results
    os.makedirs("experiments", exist_ok=True)
    out_path = f"experiments/ablation_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    return best_val_acc, test_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return total_loss / total, 100. * correct / total


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Ablation training (local, no quantum)")
    parser.add_argument("--model", required=True, choices=["classical_conv", "param_linear"],
                        help="Which ablation model to train")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--test", action="store_true", help="Quick test: 1 batch only")
    args = parser.parse_args()

    if args.model == "classical_conv":
        model = ClassicalBaselineNet(num_classes=config.NUM_CLASSES)
    else:
        model = ParamMatchedLinearNet(num_classes=config.NUM_CLASSES)

    train_loader, val_loader, test_loader = load_data()

    if args.test:
        # Quick forward pass test
        x, y = next(iter(train_loader))
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(f"Forward pass OK: {x.shape} -> {out.shape}")
        total = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total}")
        return

    start = time.time()
    best_val, test_acc = train(model, train_loader, val_loader, test_loader, args)
    elapsed = time.time() - start
    h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
    print(f"\nToplam sure: {h}s {m}dk")
    print(f"Best val: {best_val:.2f}%, Test: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
