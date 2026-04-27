#!/usr/bin/env python3
"""
Train stronger modern classical baselines under the publication benchmark stack.

Examples:
  python train_modern_baselines.py --model resnet18_cifar_gray
  python train_modern_baselines.py --model resnet18_cifar_gray --epochs 80
  python train_modern_baselines.py --model resnet18_cifar_gray --seed 42 --split-seed 42
  python train_modern_baselines.py --model resnet18_cifar_gray --test
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
    make_loader,
    result_paths,
    set_global_seed,
    summarize_model_params,
)
from src.benchmark_training import LabelSmoothingCrossEntropy, train_classifier
from src.modern_baselines import MODERN_BASELINE_REGISTRY, create_modern_baseline


RESULT_PREFIX = "publication"
CHECKPOINT_PREFIX = "best_publication"


def load_model_data(model_name: str, split_seed: int, seed: int):
    spec = MODERN_BASELINE_REGISTRY[model_name]
    train_img, train_lbl, test_img, test_lbl = load_raw_tensors(config.IMAGE_SIZE)
    split = compute_split_indices(len(train_img), config.VALIDATION_SPLIT, split_seed)
    train_ds = TensorImageDataset(
        train_img,
        train_lbl,
        indices=split.train,
        transform=spec["train_transform"],
    )
    val_ds = TensorImageDataset(train_img, train_lbl, indices=split.val)
    test_ds = TensorImageDataset(test_img, test_lbl)
    return (
        make_loader(train_ds, spec["batch_size"], shuffle=True, seed=seed),
        make_loader(val_ds, spec["batch_size"], shuffle=False, seed=seed),
        make_loader(test_ds, spec["batch_size"], shuffle=False, seed=seed),
        {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    )


def main():
    parser = argparse.ArgumentParser(description="Train modern classical baselines")
    parser.add_argument("--model", required=True, choices=sorted(MODERN_BASELINE_REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--split-seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--protocol-version", default=PROTOCOL_VERSION)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    set_global_seed(args.seed)
    spec = MODERN_BASELINE_REGISTRY[args.model]
    epochs = args.epochs or spec["default_epochs"]
    model = create_modern_baseline(args.model, num_classes=config.NUM_CLASSES)

    if args.test:
        x = torch.randn(2, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
        with torch.no_grad():
            out = model(x)
        params = summarize_model_params(model)
        print(f"Forward pass OK: {x.shape} -> {out.shape}")
        print(f"Total params: {params['total']}, Trainable: {params['trainable']}")
        return

    train_loader, val_loader, test_loader, dataset_sizes = load_model_data(
        args.model,
        split_seed=args.split_seed,
        seed=args.seed,
    )

    params = summarize_model_params(model)
    print(f"Model: {args.model}")
    print(f"Epochs: {epochs}")
    print(f"Seed / split-seed: {args.seed} / {args.split_seed}")
    print(f"Params: {params['total']} total, {params['trainable']} trainable")
    print(f"Dataset sizes: {dataset_sizes}")
    print("=" * 50)

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
            "notes": spec["notes"],
        },
    )
    results["run_id"] = paths["run_id"]

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=torch.device("cpu"),
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        result_payload=results,
        result_json_path=paths["json"],
        checkpoint_path=paths["checkpoint"],
        patience_limit=20,
        max_grad_norm=1.0,
        mixup_prob=0.25,
        mixup_alpha=0.2,
    )


if __name__ == "__main__":
    main()
