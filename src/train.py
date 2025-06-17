"""
Main training script for the Quanvolutional Neural Network.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from src import config
from src.model import QuanvNet
from src.dataset import get_dataloaders

def set_seeds(seed):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed) # PennyLane's numpy is already seeded in the notebook

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Performs one full training pass over the training data.
    """
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
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
    model = QuanvNet().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    print(f"Starting training on device: {config.DEVICE}")
    print(f"Model Architecture:\n{model}")

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")
        
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, config.DEVICE, "Validation")
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
    # Final evaluation on the test set
    print("\n--- Final Evaluation on Test Set ---")
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, config.DEVICE, "Test")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    print("\nTraining finished.")


if __name__ == '__main__':
    main() 