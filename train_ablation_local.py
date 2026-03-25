#!/usr/bin/env python3
"""
Local ablation training script with publication benchmark protocol support.

Examples:
  python train_ablation_local.py --model classical_conv --epochs 50 --seed 42 --split-seed 42
  python train_ablation_local.py --model param_linear --epochs 50 --seed 1 --split-seed 42
  python train_ablation_local.py --model non_trainable_quantum --epochs 50 --seed 42 --split-seed 42
  python train_ablation_local.py --model non_trainable_quantum --test
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import numpy as np
import pennylane as qml
import torch

from src import config
from src.ablation_models import (
    ClassicalBaselineNet,
    NonTrainableQuantumClassicalNet,
    ParamMatchedLinearNet,
)
from src.benchmark_protocol import (
    PROTOCOL_VERSION,
    TensorImageDataset,
    base_result_payload,
    build_cache_payload,
    cache_paths,
    compute_split_indices,
    get_platform_label,
    load_raw_tensors,
    make_loader,
    result_paths,
    set_global_seed,
    summarize_model_params,
)
from src.benchmark_training import LabelSmoothingCrossEntropy, train_classifier


RESULT_PREFIX = "ablation"
CHECKPOINT_PREFIX = "best_ablation"


def load_data(split_seed: int, seed: int):
    train_img, train_lbl, test_img, test_lbl = load_raw_tensors(config.IMAGE_SIZE)
    split = compute_split_indices(len(train_img), config.VALIDATION_SPLIT, split_seed)
    train_ds = TensorImageDataset(train_img, train_lbl, indices=split.train)
    val_ds = TensorImageDataset(train_img, train_lbl, indices=split.val)
    test_ds = TensorImageDataset(test_img, test_lbl)
    return (
        make_loader(train_ds, config.BATCH_SIZE, shuffle=True, seed=seed),
        make_loader(val_ds, config.BATCH_SIZE, shuffle=False, seed=seed),
        make_loader(test_ds, config.BATCH_SIZE, shuffle=False, seed=seed),
        {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    )


def precompute_quantum_features(
    n_filters: int,
    *,
    seed: int,
    protocol_version: str,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    cache_root = "experiments/quantum_cache"
    payload = build_cache_payload(
        namespace="henderson_local_ablation",
        protocol_version=protocol_version,
        image_size=config.IMAGE_SIZE,
        train_path=config.TRAIN_PATH,
        test_path=config.TEST_PATH,
        seed=seed,
        extra={
            "n_filters": n_filters,
            "n_qubits": 4,
            "patch_size": 2,
            "stride": 2,
            "encoding": "AngleEmbedding(4->4q)",
            "circuit": "Rot + CNOT_chain",
            "measurements": 4,
        },
    )
    paths = cache_paths(cache_root, payload)
    if not force_recompute and os.path.exists(paths["train_features"]):
        metadata = {}
        if os.path.exists(paths["metadata"]):
            with open(paths["metadata"], encoding="utf-8") as f:
                metadata = json.load(f)
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

    n_qubits = 4
    dev = qml.device("default.qubit", wires=n_qubits)
    rng = np.random.RandomState(seed)
    filter_weights = [
        rng.uniform(-np.pi, np.pi, (n_qubits, 3)).astype(np.float64)
        for _ in range(n_filters)
    ]

    @qml.qnode(dev, interface="numpy")
    def fixed_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        for i in range(n_qubits):
            qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    def process_images(images: np.ndarray) -> np.ndarray:
        all_features = []
        for img in images:
            patches = []
            for r in range(0, config.IMAGE_SIZE, 2):
                for c in range(0, config.IMAGE_SIZE, 2):
                    patches.append(img[r : r + 2, c : c + 2].flatten())
            patches = np.array(patches, dtype=np.float64)

            img_features = []
            for weights in filter_weights:
                filter_out = np.array([fixed_circuit(p, weights) for p in patches], dtype=np.float32)
                filter_out = filter_out.reshape(16, 16, n_qubits).transpose(2, 0, 1)
                img_features.append(filter_out)
            all_features.append(np.concatenate(img_features, axis=0))
        return np.array(all_features, dtype=np.float32)

    train_feat = process_images(train_np)
    test_feat = process_images(test_np)
    metadata = {
        "cache_payload": payload,
        "cache_key": paths["key"],
        "train_shape": list(train_feat.shape),
        "test_shape": list(test_feat.shape),
    }
    np.save(paths["train_features"], train_feat)
    np.save(paths["train_labels"], train_lbl.numpy())
    np.save(paths["test_features"], test_feat)
    np.save(paths["test_labels"], test_lbl.numpy())
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return train_feat, train_lbl.numpy(), test_feat, test_lbl.numpy(), metadata


def load_quantum_data(
    n_filters: int,
    *,
    seed: int,
    split_seed: int,
    protocol_version: str,
    force_recompute: bool = False,
):
    train_feat, train_lbl, test_feat, test_lbl, metadata = precompute_quantum_features(
        n_filters=n_filters,
        seed=seed,
        protocol_version=protocol_version,
        force_recompute=force_recompute,
    )
    train_feat_t = torch.tensor(train_feat, dtype=torch.float32)
    test_feat_t = torch.tensor(test_feat, dtype=torch.float32)
    train_lbl_t = torch.tensor(train_lbl, dtype=torch.long)
    test_lbl_t = torch.tensor(test_lbl, dtype=torch.long)

    split = compute_split_indices(len(train_feat_t), config.VALIDATION_SPLIT, split_seed)
    train_ds = TensorImageDataset(train_feat_t, train_lbl_t, indices=split.train)
    val_ds = TensorImageDataset(train_feat_t, train_lbl_t, indices=split.val)
    test_ds = TensorImageDataset(test_feat_t, test_lbl_t)
    loaders = (
        make_loader(train_ds, config.BATCH_SIZE, shuffle=True, seed=seed),
        make_loader(val_ds, config.BATCH_SIZE, shuffle=False, seed=seed),
        make_loader(test_ds, config.BATCH_SIZE, shuffle=False, seed=seed),
        {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        metadata,
    )
    return loaders


def build_model(args):
    if args.model == "classical_conv":
        return ClassicalBaselineNet(num_classes=config.NUM_CLASSES)
    if args.model == "param_linear":
        return ParamMatchedLinearNet(num_classes=config.NUM_CLASSES)
    if args.model == "non_trainable_quantum":
        return NonTrainableQuantumClassicalNet(
            in_channels=args.n_filters * 4, num_classes=config.NUM_CLASSES
        )
    raise ValueError(f"Unknown model: {args.model}")


def main():
    parser = argparse.ArgumentParser(description="Publication-oriented local ablation training")
    parser.add_argument(
        "--model",
        required=True,
        choices=["classical_conv", "param_linear", "non_trainable_quantum"],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n_filters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--split-seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--protocol-version", default=PROTOCOL_VERSION)
    parser.add_argument("--force-recompute-cache", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    set_global_seed(args.seed)
    model = build_model(args)

    if args.test:
        if args.model == "non_trainable_quantum":
            x = torch.randn(2, args.n_filters * 4, 16, 16)
        else:
            x = torch.randn(2, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
        model.eval()
        with torch.no_grad():
            out = model(x)
        params = summarize_model_params(model)
        print(f"Forward pass OK: {x.shape} -> {out.shape}")
        print(f"Total params: {params['total']}, Trainable: {params['trainable']}")
        return

    if args.model == "non_trainable_quantum":
        train_loader, val_loader, test_loader, dataset_sizes, cache_metadata = load_quantum_data(
            args.n_filters,
            seed=args.seed,
            split_seed=args.split_seed,
            protocol_version=args.protocol_version,
            force_recompute=args.force_recompute_cache,
        )
    else:
        train_loader, val_loader, test_loader, dataset_sizes = load_data(
            split_seed=args.split_seed,
            seed=args.seed,
        )
        cache_metadata = None

    params = summarize_model_params(model)
    paths = result_paths(
        args.model,
        args.seed,
        args.split_seed,
        args.protocol_version,
        result_prefix=RESULT_PREFIX,
        checkpoint_prefix=CHECKPOINT_PREFIX,
    )
    checkpoint_alias = None
    json_alias = None
    if args.seed == config.RANDOM_SEED and args.split_seed == config.RANDOM_SEED:
        checkpoint_alias = f"models/{CHECKPOINT_PREFIX}_{args.model}.pth"
        json_alias = f"experiments/{RESULT_PREFIX}_{args.model}.json"

    print(f"Model: {args.model}")
    print(f"Protocol: {args.protocol_version}")
    print(f"Seed / split-seed: {args.seed} / {args.split_seed}")
    print(f"Params: {params['total']} total, {params['trainable']} trainable")
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Platform: {get_platform_label()}")
    print("=" * 50)

    extra = {"dataset_sizes": dataset_sizes}
    if cache_metadata is not None:
        extra["cache_metadata"] = cache_metadata
        extra["n_filters"] = args.n_filters

    results = base_result_payload(
        model_name=args.model,
        source="repo-local-ablation",
        family="current-local",
        seed=args.seed,
        split_seed=args.split_seed,
        protocol_version=args.protocol_version,
        params=params,
        extra=extra,
    )
    results["run_id"] = paths["run_id"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=torch.device("cpu"),
        epochs=args.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        result_payload=results,
        result_json_path=paths["json"],
        checkpoint_path=paths["checkpoint"],
        checkpoint_alias_path=checkpoint_alias,
        json_alias_path=json_alias,
        patience_limit=15,
        max_grad_norm=1.0,
        mixup_prob=0.5,
        mixup_alpha=0.2,
    )


if __name__ == "__main__":
    main()
