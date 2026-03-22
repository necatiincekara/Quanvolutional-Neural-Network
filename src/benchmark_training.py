"""
Shared training loop helpers for publication benchmark scripts.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .benchmark_protocol import ensure_parent_dir, with_runtime, write_json


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_pred = torch.log_softmax(pred, dim=-1)
        nll_loss = -log_pred.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_pred.mean(dim=-1)
        return ((1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss).mean()


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def train_classifier(
    *,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    result_payload: Dict[str, Any],
    result_json_path: str,
    checkpoint_path: str,
    patience_limit: Optional[int] = None,
    max_grad_norm: Optional[float] = 1.0,
    mixup_prob: float = 0.0,
    mixup_alpha: float = 0.2,
    checkpoint_alias_path: Optional[str] = None,
    json_alias_path: Optional[str] = None,
) -> Dict[str, Any]:
    model = model.to(device)
    best_val_acc = -1.0
    patience_counter = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            if mixup_prob > 0 and np.random.random() < mixup_prob:
                images, y_a, y_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
                output = model(images)
                loss = lam * criterion(output, y_a) + (1 - lam) * criterion(output, y_b)
            else:
                output = model(images)
                loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / max(total, 1):.1f}%")

        if scheduler is not None:
            scheduler.step()

        train_loss = total_loss / max(total, 1)
        train_acc = 100.0 * correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        result_payload["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 2),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 2),
            }
        )

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ensure_parent_dir(checkpoint_path)
            torch.save(model.state_dict(), checkpoint_path)
            if checkpoint_alias_path is not None:
                ensure_parent_dir(checkpoint_alias_path)
                torch.save(model.state_dict(), checkpoint_alias_path)
            print(f"  NEW BEST: {val_acc:.2f}%")
        else:
            patience_counter += 1

        if patience_limit is not None and patience_counter >= patience_limit:
            print(f"  Early stopping at epoch {epoch}")
            break

    print("\nFinal test evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    result_payload["best_val_acc"] = round(best_val_acc, 2)
    result_payload["test_loss"] = round(test_loss, 4)
    result_payload["test_acc"] = round(test_acc, 2)
    with_runtime(result_payload, time.time() - start)

    write_json(result_json_path, result_payload)
    if json_alias_path is not None:
        write_json(json_alias_path, result_payload)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Results saved to {result_json_path}")
    return result_payload
