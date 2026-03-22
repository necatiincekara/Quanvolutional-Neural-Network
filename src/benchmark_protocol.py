"""
Shared benchmark protocol helpers for publication-oriented experiments.

This module centralizes:
- seed handling
- deterministic train/validation splits
- common dataset loading
- result metadata generation
- cache versioning for non-trainable quantum preprocessing
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import config
from .dataset import load_images_from_folder


PROTOCOL_VERSION = "publication_v1"


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    val: list[int]


class TensorImageDataset(Dataset):
    """Tensor-backed dataset with optional transforms and subset indices."""

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        indices: Optional[Sequence[int]] = None,
        transform=None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = list(indices) if indices is not None else list(range(len(images)))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = self.indices[idx]
        image = self.images[base_idx]
        label = self.labels[base_idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_platform_label() -> str:
    if os.path.exists("/content"):
        if torch.cuda.is_available():
            return f"colab-{torch.cuda.get_device_name(0)}"
        return "colab-cpu"
    if platform.system() == "Darwin":
        return "mac-cpu"
    return f"{platform.system().lower()}-cpu"


def load_raw_tensors(
    image_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load raw train/test tensors from repo paths."""
    size = image_size or config.IMAGE_SIZE
    train_img, train_lbl = load_images_from_folder(config.TRAIN_PATH, config.TAGS, size)
    test_img, test_lbl = load_images_from_folder(config.TEST_PATH, config.TAGS, size)

    train_img_t = torch.tensor(train_img / 255.0, dtype=torch.float32).unsqueeze(1)
    test_img_t = torch.tensor(test_img / 255.0, dtype=torch.float32).unsqueeze(1)
    train_lbl_t = torch.tensor(train_lbl, dtype=torch.long)
    test_lbl_t = torch.tensor(test_lbl, dtype=torch.long)
    return train_img_t, train_lbl_t, test_img_t, test_lbl_t


def compute_split_indices(
    dataset_size: int,
    val_fraction: float,
    split_seed: int,
) -> SplitIndices:
    """Create deterministic train/validation split indices."""
    generator = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(dataset_size, generator=generator).tolist()
    val_size = int(dataset_size * val_fraction)
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]
    return SplitIndices(train=train_indices, val=val_indices)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def build_cache_payload(
    *,
    namespace: str,
    protocol_version: str,
    image_size: int,
    train_path: str,
    test_path: str,
    seed: int,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    payload = {
        "namespace": namespace,
        "protocol_version": protocol_version,
        "image_size": image_size,
        "train_path": str(Path(train_path).resolve()),
        "test_path": str(Path(test_path).resolve()),
        "seed": seed,
    }
    payload.update(extra)
    return payload


def cache_key(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def cache_paths(cache_root: str, payload: Dict[str, Any]) -> Dict[str, str]:
    key = cache_key(payload)
    os.makedirs(cache_root, exist_ok=True)
    prefix = os.path.join(cache_root, key)
    return {
        "key": key,
        "train_features": f"{prefix}_train_features.npy",
        "train_labels": f"{prefix}_train_labels.npy",
        "test_features": f"{prefix}_test_features.npy",
        "test_labels": f"{prefix}_test_labels.npy",
        "metadata": f"{prefix}_metadata.json",
    }


def result_run_id(model_name: str, seed: int, split_seed: int, protocol_version: str) -> str:
    return f"{model_name}__{protocol_version}__seed{seed}__split{split_seed}"


def result_paths(
    model_name: str,
    seed: int,
    split_seed: int,
    protocol_version: str,
    *,
    result_prefix: str,
    checkpoint_prefix: str,
) -> Dict[str, str]:
    run_id = result_run_id(model_name, seed, split_seed, protocol_version)
    return {
        "run_id": run_id,
        "json": os.path.join("experiments", f"{result_prefix}_{model_name}_seed{seed}_split{split_seed}.json"),
        "checkpoint": os.path.join("models", f"{checkpoint_prefix}_{model_name}_seed{seed}_split{split_seed}.pth"),
    }


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def summarize_model_params(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def base_result_payload(
    *,
    model_name: str,
    source: str,
    family: str,
    seed: int,
    split_seed: int,
    protocol_version: str,
    params: Dict[str, int],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "model": model_name,
        "source": source,
        "family": family,
        "train_seed": seed,
        "split_seed": split_seed,
        "protocol_version": protocol_version,
        "platform": get_platform_label(),
        "params": params,
        "total_params": params["total"],
        "trainable_params": params["trainable"],
        "epochs": [],
    }
    if extra:
        payload.update(extra)
    return payload


def with_runtime(payload: Dict[str, Any], runtime_seconds: float) -> Dict[str, Any]:
    payload["runtime_seconds"] = round(runtime_seconds, 2)
    payload["runtime_minutes"] = round(runtime_seconds / 60.0, 2)
    return payload
