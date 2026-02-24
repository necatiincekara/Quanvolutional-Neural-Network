#!/usr/bin/env python3
"""
V7 Training Entry Point
Run on Google Colab with A100 GPU for best performance.

Usage:
    python train_v7.py                          # Default: data_reuploading, 3 epochs
    python train_v7.py --epochs 5               # More epochs
    python train_v7.py --circuit strongly_entangling
    python train_v7.py --circuit hardware_efficient
"""

import argparse
import torch
import time
import os

def main():
    parser = argparse.ArgumentParser(description="V7 Enhanced Quantum Training")
    parser.add_argument('--circuit', type=str, default='data_reuploading',
                        choices=['data_reuploading', 'strongly_entangling', 'hardware_efficient'],
                        help='Quantum circuit type')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--target', type=float, default=25.0,
                        help='Target validation accuracy (default: 25.0)')
    args = parser.parse_args()

    # Environment info
    from src import config
    print("=" * 60)
    print("V7 ENHANCED QUANTUM TRAINING")
    print("=" * 60)
    print(f"Platform:        {'Colab' if config.IS_COLAB else 'Local'}")
    print(f"Compute Device:  {config.DEVICE}")
    print(f"Quantum Device:  {config.QUANTUM_DEVICE}")
    print(f"Circuit Type:    {args.circuit}")
    print(f"Epochs:          {args.epochs}")
    print(f"Target Accuracy: {args.target}%")

    if torch.cuda.is_available():
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory:      {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print("=" * 60)

    # Mount Google Drive if on Colab
    if config.IS_COLAB:
        try:
            from google.colab import drive
            if not os.path.exists('/content/drive/MyDrive'):
                drive.mount('/content/drive')
                print("Google Drive mounted.")
        except ImportError:
            pass

    # Check dataset exists
    if not os.path.exists(config.TRAIN_PATH):
        print(f"\nERROR: Training data not found at: {config.TRAIN_PATH}")
        print("Make sure Google Drive is mounted and dataset path is correct.")
        return

    start_time = time.time()

    # Run training
    from src.enhanced_training import run_enhanced_training
    best_acc, test_acc = run_enhanced_training(
        circuit_type=args.circuit,
        num_epochs=args.epochs
    )

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print(f"\nTotal training time: {hours}h {minutes}m")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")

if __name__ == '__main__':
    main()
