"""
Faithful-as-possible reproductions of thesis-era models.

Important note:
The thesis tables for HQNN-II contain an internal mismatch between the reported
spatial shapes and the reported parameter counts. The implementation below matches:
- the reported parameter counts
- the reported 2-qubit fixed-quantum preprocessing narrative
- the reported downstream dense-layer sizes

That choice is recorded explicitly in result metadata as an implementation note.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torchvision import transforms

from . import config
from .benchmark_protocol import (
    PROTOCOL_VERSION,
    build_cache_payload,
    cache_paths,
    load_raw_tensors,
    write_json,
)


def thesis_cnniiii_transform() -> transforms.Compose:
    """Approximate thesis augmentation settings for CNN-IIII."""
    return transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=12,
                translate=(0.10, 0.10),
                scale=(0.95, 1.05),
                shear=10,
                fill=0.0,
            )
        ]
    )


class ThesisCNN3Net(nn.Module):
    def __init__(self, num_classes: int = 44):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ThesisCNNIIIINet(nn.Module):
    def __init__(self, num_classes: int = 44):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ThesisHQNN2Net(nn.Module):
    """Faithful reconstruction matching reported HQNN-II parameter count."""

    def __init__(self, num_classes: int = 44):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5),  # 16 -> 12
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 12 -> 6
            nn.Conv2d(32, 32, kernel_size=3), # 6 -> 4
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 4 -> 2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ThesisHQNN3Net(nn.Module):
    def __init__(self, num_classes: int = 44):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3),  # 16 -> 14
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 14 -> 7
            nn.Conv2d(16, 32, kernel_size=3), # 7 -> 5
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 5 -> 2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


THESIS_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "thesis_cnn3": {
        "builder": ThesisCNN3Net,
        "family": "thesis-faithful",
        "source": "thesis-faithful",
        "default_epochs": 40,
        "batch_size": 128,
        "needs_quantum_cache": False,
        "train_transform": None,
        "thesis_reference": {"val_acc": 89.50, "test_acc": 83.05},
        "notes": "CNN-III thesis reproduction candidate.",
    },
    "thesis_cnniiii": {
        "builder": ThesisCNNIIIINet,
        "family": "thesis-faithful",
        "source": "thesis-faithful",
        "default_epochs": 500,
        "batch_size": 128,
        "needs_quantum_cache": False,
        "train_transform": thesis_cnniiii_transform(),
        "thesis_reference": {"val_acc": 84.21, "test_acc": 83.69},
        "notes": "CNN-IIII best thesis classical model with augmentation.",
    },
    "thesis_hqnn2": {
        "builder": ThesisHQNN2Net,
        "family": "thesis-faithful",
        "source": "thesis-faithful",
        "default_epochs": 50,
        "batch_size": 128,
        "needs_quantum_cache": True,
        "quantum_variant": "hqnn2_non_entangled_2q",
        "thesis_reference": {"val_acc": 86.88, "test_acc": 82.40},
        "notes": (
            "2-qubit non-entangled quantum preprocessing with 2 filters producing 4 channels. "
            "Spatial implementation chosen to match the reported parameter count (248,428)."
        ),
    },
    "thesis_hqnn3": {
        "builder": ThesisHQNN3Net,
        "family": "thesis-faithful",
        "source": "thesis-faithful",
        "default_epochs": 50,
        "batch_size": 128,
        "needs_quantum_cache": True,
        "quantum_variant": "hqnn3_entangled_2q",
        "thesis_reference": {"val_acc": 87.46, "test_acc": 80.47},
        "notes": "2-qubit entangled quantum preprocessing with 2 filters producing 4 channels.",
    },
}


def create_thesis_model(model_name: str, num_classes: int = 44) -> nn.Module:
    return THESIS_MODEL_REGISTRY[model_name]["builder"](num_classes=num_classes)


def normalize_amplitude_patch(patch: np.ndarray) -> np.ndarray:
    patch = np.asarray(patch, dtype=np.float64)
    norm = np.linalg.norm(patch)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return patch / norm


def precompute_thesis_quantum_features(
    model_name: str,
    seed: int,
    *,
    protocol_version: str = PROTOCOL_VERSION,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if model_name not in {"thesis_hqnn2", "thesis_hqnn3"}:
        raise ValueError(f"{model_name} does not use thesis quantum preprocessing.")

    is_entangled = model_name == "thesis_hqnn3"
    cache_root = "experiments/quantum_cache"
    payload = build_cache_payload(
        namespace=model_name,
        protocol_version=protocol_version,
        image_size=config.IMAGE_SIZE,
        train_path=config.TRAIN_PATH,
        test_path=config.TEST_PATH,
        seed=seed,
        extra={
            "qubits": 2,
            "filters": 2,
            "patch_size": 2,
            "stride": 2,
            "encoding": "AmplitudeEmbedding(4->2q)",
            "circuit": "RY+RZ" if not is_entangled else "RY+RZ+CNOT",
            "assumption": "Quantum feature tensor is 4x16x16 (2 filters x 2 expvals).",
        },
    )
    paths = cache_paths(cache_root, payload)

    if not force_recompute and os.path.exists(paths["train_features"]):
        metadata = {}
        if os.path.exists(paths["metadata"]):
            with open(paths["metadata"], encoding="utf-8") as f:
                metadata = __import__("json").load(f)
        return (
            np.load(paths["train_features"]),
            np.load(paths["train_labels"]),
            np.load(paths["test_features"]),
            np.load(paths["test_labels"]),
            metadata,
        )

    train_img, train_lbl, test_img, test_lbl = load_raw_tensors(config.IMAGE_SIZE)
    train_np = train_img.squeeze(1).numpy()
    test_np = test_img.squeeze(1).numpy()

    rng = np.random.RandomState(seed)
    filter_weights = [rng.uniform(-np.pi, np.pi, size=(2, 2)).astype(np.float64) for _ in range(2)]
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="numpy")
    def thesis_2q_circuit(inputs, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(2), normalize=True, pad_with=0.0)
        for qubit in range(2):
            qml.RY(weights[qubit, 0], wires=qubit)
            qml.RZ(weights[qubit, 1], wires=qubit)
        if is_entangled:
            qml.CNOT(wires=[0, 1])
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    def process_images(images: np.ndarray) -> np.ndarray:
        all_features = []
        for img in images:
            patches = []
            for r in range(0, config.IMAGE_SIZE, 2):
                for c in range(0, config.IMAGE_SIZE, 2):
                    patch = img[r : r + 2, c : c + 2].flatten()
                    patches.append(normalize_amplitude_patch(patch))
            patches = np.array(patches, dtype=np.float64)

            img_features = []
            for weights in filter_weights:
                filter_out = np.array([thesis_2q_circuit(p, weights) for p in patches], dtype=np.float32)
                filter_out = filter_out.reshape(16, 16, 2).transpose(2, 0, 1)
                img_features.append(filter_out)
            all_features.append(np.concatenate(img_features, axis=0))
        return np.array(all_features, dtype=np.float32)

    train_feat = process_images(train_np)
    test_feat = process_images(test_np)
    metadata = {
        "cache_payload": payload,
        "cache_key": paths["key"],
        "notes": THESIS_MODEL_REGISTRY[model_name]["notes"],
        "train_shape": list(train_feat.shape),
        "test_shape": list(test_feat.shape),
    }
    np.save(paths["train_features"], train_feat)
    np.save(paths["train_labels"], train_lbl.numpy())
    np.save(paths["test_features"], test_feat)
    np.save(paths["test_labels"], test_lbl.numpy())
    write_json(paths["metadata"], metadata)
    return train_feat, train_lbl.numpy(), test_feat, test_lbl.numpy(), metadata
