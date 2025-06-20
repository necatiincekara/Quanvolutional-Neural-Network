"""
Main training script for the Quanvolutional Neural Network.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import argparse
from torch.amp import autocast, GradScaler
from . import config
from .model import QuanvNet
from .dataset import get_dataloaders

def set_seeds(seed):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed) # PennyLane's numpy is already seeded in the notebook

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, device):
    """
    Performs one full training pass over the training data.
    """
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Debug: print q_out.std for the first batch of each epoch
        if batch_idx == 0:
            with torch.no_grad():
                q_out = model.quanv(images[:4]).detach()
                print(f"[DEBUG] q_out std = {q_out.std():.2e}")

        scaler.scale(loss).backward()

        # Debug: print gradient mean after backward on first batch
        if batch_idx == 0:
            grad_mean = torch.mean(torch.stack([
                p.grad.abs().mean() for p in model.parameters() if p.grad is not None
            ]))
            print(f"[DEBUG] grad |mean| = {grad_mean:.2e}")

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(train_loader)

def evaluate(model, data_loader, criterion, device, data_type="Validation"):
    """
    Evaluates the model on the given dataset (validation or test).
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f"{data_type}", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            accuracy = 100 * correct_predictions / total_samples
            progress_bar.set_postfix(loss=loss.item(), accuracy=f"{accuracy:.2f}%")

    avg_loss = total_loss / len(data_loader)
    avg_accuracy = 100 * correct_predictions / total_samples
    return avg_loss, avg_accuracy

def main():
    """
    Main function to run the training and evaluation pipeline.
    """
    set_seeds(config.RANDOM_SEED)
    
    # Create device object
    device = torch.device(config.DEVICE)

    parser = argparse.ArgumentParser(description="Train or resume Quanvolutional NN")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    args = parser.parse_args()

    # directories and checkpoint paths
    os.makedirs("models", exist_ok=True)
    best_model_path = "models/best_quanv_net.pth"
    ckpt_path = "models/checkpoint_latest.pth"
    best_val_accuracy = 0.0

    # Get data loaders
    # Note: This will fail if the Drive path is not mounted in the environment.
    # In Colab, you must mount your Google Drive first.
    # On local, you need to have the data in the specified path or change the path in config.py
    try:
        train_loader, val_loader, test_loader = get_dataloaders()
    except FileNotFoundError as e:
        print("="*50)
        print("ERROR: Data directory not found.")
        print(f"Please check that the path in 'src/config.py' is correct: {e}")
        print("If running in Google Colab, ensure you have mounted your Google Drive.")
        print("Example Colab code to mount drive:")
        print("from google.colab import drive")
        print("drive.mount('/content/drive')")
        print("="*50)
        return

    # Initialize model, loss, and optimizer
    model = QuanvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Warm-up then cosine schedule
    total_steps = len(train_loader) * config.NUM_EPOCHS
    warmup_steps = 50
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # Use torch.cos for tensor operations
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535, device=optimizer.param_groups[0]['params'][0].device)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=device.type=="cuda")

    start_epoch = 0
    if args.resume and os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_accuracy = checkpoint.get("best_val_accuracy", 0.0)

    print(f"Starting training on device: {device}")
    print(f"Model Architecture:\n{model}")

    # Training loop
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, device)
        print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")
        
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, "Validation")
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        # Save the model if it has the best validation accuracy so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ New best model saved with accuracy: {val_accuracy:.2f}%")

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_accuracy": best_val_accuracy,
        }, ckpt_path)
        
    # Final evaluation on the test set
    print("\n--- Final Evaluation on Test Set ---")
    # Load the best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))
    print("Loaded best model for final testing.")
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device, "Test")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    print("\nTraining finished.")


if __name__ == '__main__':
    main() 