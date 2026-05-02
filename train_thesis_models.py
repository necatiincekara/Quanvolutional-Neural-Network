#!/usr/bin/env python3
"""
Train thesis-faithful reproduction models under the publication benchmark protocol.

Examples:
  python train_thesis_models.py --model thesis_cnn3
  python train_thesis_models.py --model thesis_cnniiii --epochs 100
  python train_thesis_models.py --model thesis_hqnn2 --seed 42 --split-seed 42
  python train_thesis_models.py --model thesis_hqnn2 --train-fraction 0.10 --protocol-version low_data_pilot_v1
  python train_thesis_models.py --model thesis_hqnn2 --test
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from src import config
from src.benchmark_protocol import (
    PROTOCOL_VERSION,
    TensorImageDataset,
    base_result_payload,
    compute_split_indices,
    load_raw_tensors,
    low_data_result_paths,
    make_loader,
    result_paths,
    set_global_seed,
    stratified_train_subset_indices,
    summarize_model_params,
)
from src.benchmark_training import train_classifier
from src.thesis_models import (
    THESIS_MODEL_REGISTRY,
    create_thesis_model,
    precompute_thesis_quantum_features,
)


RESULT_PREFIX = "publication"
CHECKPOINT_PREFIX = "best_publication"


def load_raw_model_data(model_name: str, split_seed: int, seed: int, train_fraction: float, fraction_seed: int):
    spec = THESIS_MODEL_REGISTRY[model_name]
    train_img, train_lbl, test_img, test_lbl = load_raw_tensors(config.IMAGE_SIZE)
    split = compute_split_indices(len(train_img), config.VALIDATION_SPLIT, split_seed)
    selection = stratified_train_subset_indices(split.train, train_lbl, train_fraction, fraction_seed)
    train_ds = TensorImageDataset(
        train_img,
        train_lbl,
        indices=selection.indices,
        transform=spec["train_transform"],
    )
    val_ds = TensorImageDataset(train_img, train_lbl, indices=split.val)
    test_ds = TensorImageDataset(test_img, test_lbl)
    return (
        make_loader(train_ds, spec["batch_size"], shuffle=True, seed=seed),
        make_loader(val_ds, spec["batch_size"], shuffle=False, seed=seed),
        make_loader(test_ds, spec["batch_size"], shuffle=False, seed=seed),
        {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        selection.metadata,
        None,
    )


def load_quantum_model_data(
    model_name: str,
    split_seed: int,
    seed: int,
    train_fraction: float,
    fraction_seed: int,
    protocol_version: str,
    force_recompute: bool,
):
    spec = THESIS_MODEL_REGISTRY[model_name]
    train_feat, train_lbl, test_feat, test_lbl, cache_metadata = precompute_thesis_quantum_features(
        model_name,
        seed=seed,
        protocol_version=protocol_version,
        force_recompute=force_recompute,
    )
    train_feat_t = torch.tensor(train_feat, dtype=torch.float32)
    test_feat_t = torch.tensor(test_feat, dtype=torch.float32)
    train_lbl_t = torch.tensor(train_lbl, dtype=torch.long)
    test_lbl_t = torch.tensor(test_lbl, dtype=torch.long)
    split = compute_split_indices(len(train_feat_t), config.VALIDATION_SPLIT, split_seed)
    selection = stratified_train_subset_indices(split.train, train_lbl_t, train_fraction, fraction_seed)
    train_ds = TensorImageDataset(train_feat_t, train_lbl_t, indices=selection.indices)
    val_ds = TensorImageDataset(train_feat_t, train_lbl_t, indices=split.val)
    test_ds = TensorImageDataset(test_feat_t, test_lbl_t)
    return (
        make_loader(train_ds, spec["batch_size"], shuffle=True, seed=seed),
        make_loader(val_ds, spec["batch_size"], shuffle=False, seed=seed),
        make_loader(test_ds, spec["batch_size"], shuffle=False, seed=seed),
        {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        selection.metadata,
        cache_metadata,
    )


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main():
    parser = argparse.ArgumentParser(description="Train thesis-faithful reproduction models")
    parser.add_argument("--model", required=True, choices=sorted(THESIS_MODEL_REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--split-seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--fraction-seed", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "auto"], default="cpu")
    parser.add_argument("--protocol-version", default=PROTOCOL_VERSION)
    parser.add_argument("--force-recompute-cache", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.fraction_seed is None:
        args.fraction_seed = args.split_seed
    is_low_data_run = args.train_fraction < 1.0 or args.protocol_version.startswith("low_data")
    cache_protocol_version = PROTOCOL_VERSION if is_low_data_run else args.protocol_version

    set_global_seed(args.seed)
    spec = THESIS_MODEL_REGISTRY[args.model]
    epochs = args.epochs or spec["default_epochs"]
    model = create_thesis_model(args.model, num_classes=config.NUM_CLASSES)
    device = select_device(args.device)

    if args.test:
        if spec["needs_quantum_cache"]:
            x = torch.randn(2, 4, 16, 16)
        else:
            x = torch.randn(2, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
        with torch.no_grad():
            out = model(x)
        params = summarize_model_params(model)
        print(f"Forward pass OK: {x.shape} -> {out.shape}")
        print(f"Total params: {params['total']}, Trainable: {params['trainable']}")
        return

    if spec["needs_quantum_cache"]:
        train_loader, val_loader, test_loader, dataset_sizes, low_data_metadata, cache_metadata = load_quantum_model_data(
            args.model,
            split_seed=args.split_seed,
            seed=args.seed,
            train_fraction=args.train_fraction,
            fraction_seed=args.fraction_seed,
            protocol_version=cache_protocol_version,
            force_recompute=args.force_recompute_cache,
        )
    else:
        train_loader, val_loader, test_loader, dataset_sizes, low_data_metadata, cache_metadata = load_raw_model_data(
            args.model,
            split_seed=args.split_seed,
            seed=args.seed,
            train_fraction=args.train_fraction,
            fraction_seed=args.fraction_seed,
        )

    params = summarize_model_params(model)
    print(f"Model: {args.model}")
    print(f"Epochs: {epochs}")
    print(f"Seed / split-seed: {args.seed} / {args.split_seed}")
    print(f"Train fraction / fraction-seed: {args.train_fraction:.2f} / {args.fraction_seed}")
    print(f"Params: {params['total']} total, {params['trainable']} trainable")
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Device: {device}")
    print("=" * 50)

    if is_low_data_run:
        paths = low_data_result_paths(
            args.model,
            args.seed,
            args.split_seed,
            args.fraction_seed,
            args.train_fraction,
            args.protocol_version,
            result_prefix=RESULT_PREFIX,
            checkpoint_prefix=CHECKPOINT_PREFIX,
        )
    else:
        paths = result_paths(
            args.model,
            args.seed,
            args.split_seed,
            args.protocol_version,
            result_prefix=RESULT_PREFIX,
            checkpoint_prefix=CHECKPOINT_PREFIX,
        )
    results = base_result_payload(
        model_name=args.model,
        source=spec["source"],
        family=spec["family"],
        seed=args.seed,
        split_seed=args.split_seed,
        protocol_version=args.protocol_version,
        params=params,
        extra={
            "dataset_sizes": dataset_sizes,
            "benchmark_axis": "low-data-scaling" if is_low_data_run else "publication",
            "train_fraction": args.train_fraction,
            "fraction_seed": args.fraction_seed,
            "low_data": low_data_metadata,
            "thesis_reference": spec["thesis_reference"],
            "notes": spec["notes"],
            "cache_metadata": cache_metadata,
        },
    )
    results["run_id"] = paths["run_id"]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        optimizer=optimizer,
        scheduler=None,
        criterion=criterion,
        result_payload=results,
        result_json_path=paths["json"],
        checkpoint_path=paths["checkpoint"],
        patience_limit=None,
        max_grad_norm=None,
        mixup_prob=0.0,
        mixup_alpha=0.0,
    )


if __name__ == "__main__":
    main()
