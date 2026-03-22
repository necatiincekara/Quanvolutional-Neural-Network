#!/usr/bin/env python3
"""
Local ablation training script for M4 Mac Mini.
Runs classical-only models and Henderson-style non-trainable quantum.

Usage:
  python train_ablation_local.py --model classical_conv --epochs 50
  python train_ablation_local.py --model param_linear --epochs 50
  python train_ablation_local.py --model non_trainable_quantum --epochs 50
  python train_ablation_local.py --model non_trainable_quantum --epochs 1 --test
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

from src.ablation_models import (
    ClassicalBaselineNet,
    ParamMatchedLinearNet,
    NonTrainableQuantumClassicalNet,
)
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


# ---- Henderson-Style Quantum Pre-computation ----

def precompute_quantum_features(n_filters=4):
    """
    Henderson et al. (2020) approach: apply fixed random quantum circuits
    to raw images ONCE, cache results to disk.

    Each filter: 4-qubit circuit with random fixed Rot gates + CNOT chain.
    Processes 2x2 patches (stride 2) on 32x32 images -> 16x16 output.
    4 filters x 4 qubits = 16 output channels.
    """
    import pennylane as qml

    cache_dir = "experiments/quantum_cache"
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = f"{cache_dir}/train_features_{n_filters}f.npy"
    if os.path.exists(cache_file):
        print(f"Loading cached quantum features from {cache_dir}/...")
        train_feat = np.load(f"{cache_dir}/train_features_{n_filters}f.npy")
        train_lbl = np.load(f"{cache_dir}/train_labels.npy")
        test_feat = np.load(f"{cache_dir}/test_features_{n_filters}f.npy")
        test_lbl = np.load(f"{cache_dir}/test_labels.npy")
        print(f"  Train: {train_feat.shape}, Test: {test_feat.shape}")
        return train_feat, train_lbl, test_feat, test_lbl

    print("=" * 50)
    print("Henderson-style quantum pre-computation")
    print(f"  Filters: {n_filters}, Qubits: 4, Patches per image: 256")
    print("=" * 50)

    # Load raw images
    from src.dataset import load_images_from_folder
    train_img, train_lbl = load_images_from_folder(
        config.TRAIN_PATH, config.TAGS, config.IMAGE_SIZE
    )
    test_img, test_lbl = load_images_from_folder(
        config.TEST_PATH, config.TAGS, config.IMAGE_SIZE
    )
    train_img = train_img / 255.0
    test_img = test_img / 255.0

    # Create quantum circuit (non-trainable, fixed random params)
    n_qubits = 4
    dev = qml.device('default.qubit', wires=n_qubits)

    # Generate reproducible random weights for each filter
    rng = np.random.RandomState(42)
    filter_weights = [
        rng.uniform(-np.pi, np.pi, (n_qubits, 3)).astype(np.float64)
        for _ in range(n_filters)
    ]

    @qml.qnode(dev, interface='numpy')
    def fixed_circuit(inputs, weights):
        """Non-trainable 4-qubit circuit: AngleEmbedding + Rot + CNOT chain."""
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        for i in range(n_qubits):
            qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    def process_images(images, desc="Processing"):
        """Apply all quantum filters to all images."""
        all_features = []
        for idx in tqdm(range(len(images)), desc=desc):
            img = images[idx]  # (32, 32)

            # Extract 2x2 patches with stride 2 -> (16, 16) grid of patches
            # Each patch: 4 pixel values (2x2)
            patches = []
            for r in range(0, 32, 2):
                for c in range(0, 32, 2):
                    patch = img[r:r+2, c:c+2].flatten()  # (4,)
                    patches.append(patch)
            patches = np.array(patches)  # (256, 4)

            # Apply each quantum filter
            img_features = []
            for f_idx in range(n_filters):
                w = filter_weights[f_idx]
                filter_out = np.array([
                    fixed_circuit(p, w) for p in patches
                ])  # (256, 4)
                # Reshape to (4, 16, 16) — 4 channels from 4 qubit measurements
                filter_out = filter_out.reshape(16, 16, n_qubits).transpose(2, 0, 1)
                img_features.append(filter_out)

            # Stack filters: (n_filters*4, 16, 16)
            img_features = np.concatenate(img_features, axis=0)
            all_features.append(img_features)

        return np.array(all_features, dtype=np.float32)

    t0 = time.time()
    print(f"\nPre-computing train set ({len(train_img)} images)...")
    train_feat = process_images(train_img, "Train quantum features")

    print(f"\nPre-computing test set ({len(test_img)} images)...")
    test_feat = process_images(test_img, "Test quantum features")

    elapsed = time.time() - t0
    print(f"\nPre-computation done in {elapsed/60:.1f} minutes")
    print(f"  Train features: {train_feat.shape}")
    print(f"  Test features: {test_feat.shape}")

    # Cache to disk
    np.save(f"{cache_dir}/train_features_{n_filters}f.npy", train_feat)
    np.save(f"{cache_dir}/train_labels.npy", train_lbl)
    np.save(f"{cache_dir}/test_features_{n_filters}f.npy", test_feat)
    np.save(f"{cache_dir}/test_labels.npy", test_lbl)
    print(f"Cached to {cache_dir}/")

    # Save circuit info for reproducibility
    circuit_info = {
        "n_filters": n_filters,
        "n_qubits": n_qubits,
        "circuit": "AngleEmbedding + Rot + CNOT_chain",
        "seed": 42,
        "patch_size": 2,
        "stride": 2,
        "filter_params": [w.tolist() for w in filter_weights],
    }
    with open(f"{cache_dir}/circuit_info.json", "w") as f:
        json.dump(circuit_info, f, indent=2)

    return train_feat, train_lbl, test_feat, test_lbl


def load_quantum_data(n_filters=4):
    """Load pre-computed quantum features and create DataLoaders."""
    train_feat, train_lbl, test_feat, test_lbl = precompute_quantum_features(n_filters)

    train_feat = torch.tensor(train_feat, dtype=torch.float32)
    test_feat = torch.tensor(test_feat, dtype=torch.float32)
    train_lbl = torch.tensor(train_lbl, dtype=torch.long)
    test_lbl = torch.tensor(test_lbl, dtype=torch.long)

    train_ds = TensorDataset(train_feat, train_lbl)
    test_ds = TensorDataset(test_feat, test_lbl)

    val_size = int(config.VALIDATION_SPLIT * len(train_ds))
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(
        train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED),
    )

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
    parser = argparse.ArgumentParser(description="Ablation training (local)")
    parser.add_argument("--model", required=True,
                        choices=["classical_conv", "param_linear", "non_trainable_quantum"],
                        help="Which ablation model to train")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--n_filters", type=int, default=4,
                        help="Number of quantum filters for non_trainable_quantum (default: 4)")
    parser.add_argument("--test", action="store_true", help="Quick test: 1 batch only")
    args = parser.parse_args()

    if args.model == "classical_conv":
        model = ClassicalBaselineNet(num_classes=config.NUM_CLASSES)
        if not args.test:
            train_loader, val_loader, test_loader = load_data()
    elif args.model == "param_linear":
        model = ParamMatchedLinearNet(num_classes=config.NUM_CLASSES)
        if not args.test:
            train_loader, val_loader, test_loader = load_data()
    elif args.model == "non_trainable_quantum":
        # Henderson-style: pre-compute quantum features, then train classical
        in_channels = args.n_filters * 4
        model = NonTrainableQuantumClassicalNet(
            in_channels=in_channels, num_classes=config.NUM_CLASSES
        )
        if not args.test:
            train_loader, val_loader, test_loader = load_quantum_data(args.n_filters)

    if args.test:
        # Quick forward pass test (use dummy data for quantum model to skip pre-compute)
        if args.model == "non_trainable_quantum":
            in_ch = args.n_filters * 4
            x = torch.randn(2, in_ch, 16, 16)
            y = torch.randint(0, config.NUM_CLASSES, (2,))
        else:
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
