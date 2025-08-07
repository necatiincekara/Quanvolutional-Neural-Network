"""
Enhanced Training Pipeline for Trainable Quantum-Classical Hybrid Model
Targets 90% accuracy improvement from 82% baseline with fixed quantum layers
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

from . import config
from .trainable_quantum_model import create_enhanced_model
from .dataset import get_dataloaders

class QuantumAwareOptimizer:
    """
    Custom optimizer wrapper that handles quantum and classical parameters separately
    Critical for achieving 90% accuracy target
    """
    def __init__(self, model, quantum_lr=0.001, classical_lr=0.005, quantum_momentum=0.9):
        # Separate parameters
        self.quantum_params = []
        self.classical_params = []
        
        for name, param in model.named_parameters():
            if 'quanv' in name or 'quantum' in name:
                self.quantum_params.append(param)
            else:
                self.classical_params.append(param)
        
        # Create separate optimizers
        self.quantum_optimizer = optim.Adam(
            self.quantum_params,
            lr=quantum_lr,
            betas=(quantum_momentum, 0.999),
            eps=1e-8,
            weight_decay=1e-5
        )
        
        self.classical_optimizer = optim.AdamW(
            self.classical_params,
            lr=classical_lr,
            weight_decay=1e-4
        )
        
        # Learning rate schedulers
        self.quantum_scheduler = CosineAnnealingWarmRestarts(
            self.quantum_optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        self.classical_scheduler = optim.lr_scheduler.OneCycleLR(
            self.classical_optimizer,
            max_lr=classical_lr,
            total_steps=1000,
            pct_start=0.1
        )
    
    def zero_grad(self):
        self.quantum_optimizer.zero_grad()
        self.classical_optimizer.zero_grad()
    
    def step(self):
        # Gradient clipping for quantum parameters
        torch.nn.utils.clip_grad_norm_(self.quantum_params, max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.classical_params, max_norm=1.0)
        
        self.quantum_optimizer.step()
        self.classical_optimizer.step()
    
    def scheduler_step(self):
        self.quantum_scheduler.step()
        self.classical_scheduler.step()

class GradientMonitor:
    """
    Monitors gradient flow through quantum and classical layers
    Essential for diagnosing training issues
    """
    def __init__(self, model):
        self.model = model
        self.gradient_history = {
            'quantum': [],
            'classical': []
        }
    
    def log_gradients(self):
        quantum_grads = []
        classical_grads = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if 'quanv' in name or 'quantum' in name:
                    quantum_grads.append(grad_norm)
                else:
                    classical_grads.append(grad_norm)
        
        if quantum_grads:
            self.gradient_history['quantum'].append(np.mean(quantum_grads))
        if classical_grads:
            self.gradient_history['classical'].append(np.mean(classical_grads))
    
    def check_gradient_health(self):
        """Check for vanishing or exploding gradients"""
        if len(self.gradient_history['quantum']) > 0:
            recent_quantum = np.mean(self.gradient_history['quantum'][-10:])
            if recent_quantum < 1e-6:
                print("⚠️ Warning: Quantum gradients may be vanishing")
            elif recent_quantum > 10:
                print("⚠️ Warning: Quantum gradients may be exploding")
        
        return self.gradient_history

class EnhancedTrainer:
    """
    Advanced training logic for quantum-classical hybrid model
    Implements strategies to achieve 90% accuracy from 82% baseline
    """
    def __init__(self, model, device, experiment_name="enhanced_quantum"):
        self.model = model.to(device)
        self.device = device
        self.experiment_name = experiment_name
        
        # Create experiment directory
        self.exp_dir = f"experiments/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Initialize components
        self.optimizer = QuantumAwareOptimizer(model)
        self.gradient_monitor = GradientMonitor(model)
        self.scaler = GradScaler(enabled=device.type == "cuda")
        
        # Loss with label smoothing for better generalization
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'quantum_grads': [],
            'classical_grads': []
        }
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def train_epoch(self, train_loader, epoch):
        """
        Single epoch training with enhanced techniques
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Apply mixup augmentation (50% chance)
            if np.random.random() > 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels)
            else:
                labels_a, labels_b, lam = labels, labels, 1.0
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                outputs = self.model(images)
                loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Log gradients periodically
            if batch_idx % 10 == 0:
                self.gradient_monitor.log_gradients()
            
            # Optimizer step
            self.scaler.unscale_(self.optimizer.quantum_optimizer)
            self.scaler.unscale_(self.optimizer.classical_optimizer)
            self.optimizer.step()
            self.scaler.update()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels_a).sum().item() * lam + predicted.eq(labels_b).sum().item() * (1 - lam)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
        
        # Step schedulers
        self.optimizer.scheduler_step()
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader):
        """
        Validation with comprehensive metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(val_loader), 100. * correct / total
    
    def train(self, train_loader, val_loader, num_epochs=100, target_accuracy=90.0):
        """
        Full training loop with early stopping and checkpointing
        """
        print(f"Starting training: Target accuracy = {target_accuracy}%")
        print(f"Baseline accuracy = 82% (fixed quantum layer)")
        print(f"Experiment directory: {self.exp_dir}")
        
        for epoch in range(1, num_epochs + 1):
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_loss, val_acc = self.validate(val_loader)
            
            # Check gradient health
            self.gradient_monitor.check_gradient_health()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Log progress
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Best Val Acc: {self.best_val_acc:.2f}%")
            print(f"Progress toward target: {val_acc:.2f}% / {target_accuracy}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_acc)
                print(f"✨ New best model saved: {val_acc:.2f}%")
                
                # Check if target reached
                if val_acc >= target_accuracy:
                    print(f"🎉 Target accuracy {target_accuracy}% achieved!")
                    break
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= 15:
                print("Early stopping triggered")
                break
            
            # Save training curves periodically
            if epoch % 5 == 0:
                self.plot_training_curves()
        
        # Final summary
        self.print_final_summary()
        return self.best_val_acc
    
    def save_checkpoint(self, epoch, val_acc):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_quantum': self.optimizer.quantum_optimizer.state_dict(),
            'optimizer_classical': self.optimizer.classical_optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        
        path = os.path.join(self.exp_dir, f'checkpoint_acc_{val_acc:.2f}.pth')
        torch.save(checkpoint, path)
        
        # Save metadata
        metadata = {
            'experiment_name': self.experiment_name,
            'epoch': epoch,
            'val_accuracy': val_acc,
            'baseline_accuracy': 82.0,
            'improvement': val_acc - 82.0,
            'target_accuracy': 90.0,
            'progress_percentage': (val_acc - 82.0) / (90.0 - 82.0) * 100
        }
        
        with open(os.path.join(self.exp_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def plot_training_curves(self):
        """Generate training curve plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title('Loss Curves')
        
        # Accuracy curves
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].axhline(y=82, color='r', linestyle='--', label='Baseline (82%)')
        axes[0, 1].axhline(y=90, color='g', linestyle='--', label='Target (90%)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].set_title('Accuracy Curves')
        
        # Gradient norms
        if self.gradient_monitor.gradient_history['quantum']:
            axes[1, 0].plot(self.gradient_monitor.gradient_history['quantum'], label='Quantum')
            axes[1, 0].plot(self.gradient_monitor.gradient_history['classical'], label='Classical')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
            axes[1, 0].set_title('Gradient Flow')
        
        # Improvement tracking
        improvements = [acc - 82.0 for acc in self.history['val_acc']]
        axes[1, 1].plot(improvements)
        axes[1, 1].axhline(y=8, color='g', linestyle='--', label='Target Improvement (8%)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Improvement over Baseline (%)')
        axes[1, 1].legend()
        axes[1, 1].set_title('Improvement Tracking')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, 'training_curves.png'))
        plt.close()
    
    def print_final_summary(self):
        """Print final training summary"""
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Baseline Accuracy (Fixed Quantum): 82.00%")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"Improvement: {self.best_val_acc - 82.0:.2f}%")
        print(f"Target Achievement: {(self.best_val_acc - 82.0) / 8.0 * 100:.1f}% of goal")
        print(f"Experiment saved to: {self.exp_dir}")
        print("="*50)

# -----------------
# Utility Functions
# -----------------

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_pred = torch.log_softmax(pred, dim=-1)
        loss = -log_pred.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_pred.mean(dim=-1)
        return (1 - self.smoothing) * loss + self.smoothing * smooth_loss

def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def run_enhanced_training(circuit_type='data_reuploading', num_epochs=100):
    """
    Main function to run enhanced training
    
    Args:
        circuit_type: Type of quantum circuit to use
        num_epochs: Maximum number of training epochs
    
    Returns:
        Best validation accuracy achieved
    """
    # Set device
    device = torch.device(config.DEVICE)
    
    # Load data
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # Create model
    model = create_enhanced_model(circuit_type=circuit_type, num_classes=config.NUM_CLASSES)
    
    # Create trainer
    trainer = EnhancedTrainer(model, device, experiment_name=f"quantum_{circuit_type}")
    
    # Train model
    best_acc = trainer.train(train_loader, val_loader, num_epochs=num_epochs, target_accuracy=90.0)
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    return best_acc, test_acc