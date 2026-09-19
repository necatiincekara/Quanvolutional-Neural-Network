"""
Enhanced Training Pipeline for V7 Trainable Quantum-Classical Hybrid Model
Separate quantum/classical optimizers, gradient monitoring, label smoothing, mixup
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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from . import config
from .trainable_quantum_model import create_enhanced_model
from .dataset import get_dataloaders


class QuantumAwareOptimizer:
    """
    Custom optimizer wrapper that handles quantum and classical parameters separately.
    Quantum parameters need lower LR and gentler gradient clipping.
    """
    def __init__(self, model, train_loader, num_epochs,
                 quantum_lr=0.001, classical_lr=0.005):
        # Separate quantum vs classical parameters
        self.quantum_params = []
        self.classical_params = []

        for name, param in model.named_parameters():
            if 'quanv' in name or 'quantum' in name or 'gradient_scale' in name:
                self.quantum_params.append(param)
            else:
                self.classical_params.append(param)

        print(f"Quantum params: {sum(p.numel() for p in self.quantum_params)}")
        print(f"Classical params: {sum(p.numel() for p in self.classical_params)}")

        # Quantum optimizer: Adam with conservative settings
        self.quantum_optimizer = optim.Adam(
            self.quantum_params,
            lr=quantum_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-5
        )

        # Classical optimizer: AdamW with standard settings
        self.classical_optimizer = optim.AdamW(
            self.classical_params,
            lr=classical_lr,
            weight_decay=1e-4
        )

        # Compute total training steps for schedulers
        total_steps = len(train_loader) * num_epochs

        # Quantum: cosine annealing with warm restarts
        self.quantum_scheduler = CosineAnnealingWarmRestarts(
            self.quantum_optimizer,
            T_0=max(10, num_epochs // 5),
            T_mult=2,
            eta_min=1e-6
        )

        # Classical: cosine annealing (safe, no total_steps overflow)
        self.classical_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.classical_optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )

    def zero_grad(self):
        self.quantum_optimizer.zero_grad()
        self.classical_optimizer.zero_grad()

    def step(self):
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.quantum_params, max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.classical_params, max_norm=1.0)

        self.quantum_optimizer.step()
        self.classical_optimizer.step()

    def scheduler_step(self, epoch):
        self.quantum_scheduler.step(epoch)
        self.classical_scheduler.step()


class GradientMonitor:
    """Monitors gradient flow through quantum and classical layers"""
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
                if 'quanv' in name or 'quantum' in name or 'gradient_scale' in name:
                    quantum_grads.append(grad_norm)
                else:
                    classical_grads.append(grad_norm)

        if quantum_grads:
            self.gradient_history['quantum'].append(np.mean(quantum_grads))
        if classical_grads:
            self.gradient_history['classical'].append(np.mean(classical_grads))

    def check_gradient_health(self):
        """Check for vanishing or exploding gradients"""
        issues = []
        if len(self.gradient_history['quantum']) > 0:
            recent_quantum = np.mean(self.gradient_history['quantum'][-10:])
            if recent_quantum < 1e-6:
                issues.append(f"Quantum gradients vanishing: {recent_quantum:.2e}")
            elif recent_quantum > 10:
                issues.append(f"Quantum gradients exploding: {recent_quantum:.2e}")

        if len(self.gradient_history['classical']) > 0:
            recent_classical = np.mean(self.gradient_history['classical'][-10:])
            if recent_classical < 1e-6:
                issues.append(f"Classical gradients vanishing: {recent_classical:.2e}")

        for issue in issues:
            print(f"  WARNING: {issue}")

        return self.gradient_history


class EnhancedTrainer:
    """
    V7 training pipeline with:
    - Separate quantum/classical optimizers
    - Gradient monitoring
    - Label smoothing
    - Mixup augmentation
    - Early stopping
    - Checkpoint management
    """
    def __init__(self, model, train_loader, val_loader, device,
                 num_epochs=50, experiment_name="v7_enhanced",
                 drive_backup_path=None):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.experiment_name = experiment_name
        self.drive_backup_path = drive_backup_path

        # Create experiment directory
        self.exp_dir = f"experiments/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)

        if drive_backup_path:
            os.makedirs(drive_backup_path, exist_ok=True)

        # Initialize optimizer with correct total_steps
        self.optimizer = QuantumAwareOptimizer(
            model, train_loader, num_epochs,
            quantum_lr=0.001, classical_lr=0.005
        )
        self.gradient_monitor = GradientMonitor(model)
        self.scaler = GradScaler(enabled=device.type == "cuda")

        # Label smoothing loss
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
        }

        self.best_val_acc = 0.0
        self.patience_counter = 0

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)

            # Mixup augmentation (50% chance)
            if np.random.random() > 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels)
            else:
                labels_a, labels_b, lam = labels, labels, 1.0

            self.optimizer.zero_grad()

            # Forward pass
            with autocast(device_type=self.device.type, dtype=torch.float16,
                          enabled=self.device.type == "cuda"):
                outputs = self.model(images)
                loss = lam * self.criterion(outputs, labels_a) + \
                       (1 - lam) * self.criterion(outputs, labels_b)

            # Backward pass
            self.scaler.scale(loss).backward()

            # Debug: log gradient info on first batch
            if batch_idx == 0:
                self._log_debug_info(epoch)

            # Log gradients periodically
            if batch_idx % 10 == 0:
                self.gradient_monitor.log_gradients()

            # Unscale before clipping, then step
            self.scaler.unscale_(self.optimizer.quantum_optimizer)
            self.scaler.unscale_(self.optimizer.classical_optimizer)
            self.optimizer.step()
            self.scaler.update()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted.eq(labels_a).sum().item() * lam +
                        predicted.eq(labels_b).sum().item() * (1 - lam))

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })

        # Step epoch-level schedulers
        self.optimizer.scheduler_step(epoch)

        return total_loss / len(train_loader), 100. * correct / total

    def _log_debug_info(self, epoch):
        """Log quantum layer diagnostics on first batch of each epoch"""
        # Quantum output statistics
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'gradient_scale' in name:
                    print(f"  [DEBUG] gradient_scale = {param.item():.4f}")

        # Gradient statistics
        q_grads, c_grads = [], []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                if 'quanv' in name or 'quantum' in name:
                    q_grads.append(norm)
                else:
                    c_grads.append(norm)

        if q_grads:
            print(f"  [DEBUG] quantum grad mean={np.mean(q_grads):.2e}, "
                  f"max={max(q_grads):.2e}")
        if c_grads:
            print(f"  [DEBUG] classical grad mean={np.mean(c_grads):.2e}, "
                  f"max={max(c_grads):.2e}")

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)

                with autocast(device_type=self.device.type, dtype=torch.float16,
                              enabled=self.device.type == "cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return total_loss / len(val_loader), 100. * correct / total

    def save_latest_checkpoint(self, epoch, val_acc):
        """Save latest checkpoint for resume (every epoch), also to Drive if configured"""
        import shutil
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_quantum': self.optimizer.quantum_optimizer.state_dict(),
            'optimizer_classical': self.optimizer.classical_optimizer.state_dict(),
            'scheduler_quantum': self.optimizer.quantum_scheduler.state_dict(),
            'scheduler_classical': self.optimizer.classical_scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'patience_counter': self.patience_counter,
            'history': self.history
        }
        path = 'models/checkpoint_latest_v7.pth'
        torch.save(checkpoint, path)

        # Also backup to Drive (survives runtime restart)
        if self.drive_backup_path:
            drive_ckpt = os.path.join(self.drive_backup_path, 'checkpoint_latest_v7.pth')
            shutil.copy(path, drive_ckpt)
            print(f"  [CHECKPOINT] Epoch {epoch} -> local + Drive")
        else:
            print(f"  [CHECKPOINT] Epoch {epoch} -> {path}")

    def load_checkpoint(self, checkpoint_path='models/checkpoint_latest_v7.pth'):
        """Load checkpoint for resuming training. Checks Drive backup if local not found."""
        # Try local first, then Drive backup
        if not os.path.exists(checkpoint_path) and self.drive_backup_path:
            drive_ckpt = os.path.join(self.drive_backup_path, 'checkpoint_latest_v7.pth')
            if os.path.exists(drive_ckpt):
                import shutil
                shutil.copy(drive_ckpt, checkpoint_path)
                print(f"  Restored checkpoint from Drive: {drive_ckpt}")

        if not os.path.exists(checkpoint_path):
            print("  No checkpoint found, starting from scratch.")
            return 0

        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.quantum_optimizer.load_state_dict(checkpoint['optimizer_quantum'])
        self.optimizer.classical_optimizer.load_state_dict(checkpoint['optimizer_classical'])

        if 'scheduler_quantum' in checkpoint:
            self.optimizer.quantum_scheduler.load_state_dict(checkpoint['scheduler_quantum'])
        if 'scheduler_classical' in checkpoint:
            self.optimizer.classical_scheduler.load_state_dict(checkpoint['scheduler_classical'])
        if 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])

        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.patience_counter = checkpoint.get('patience_counter', 0)
        self.history = checkpoint.get('history', self.history)
        start_epoch = checkpoint['epoch']

        print(f"  Resumed at epoch {start_epoch}, best_val_acc={self.best_val_acc:.2f}%")
        return start_epoch

    def train(self, train_loader, val_loader, target_accuracy=25.0, resume=False):
        """Full training loop with optional resume"""
        start_epoch = 0
        if resume:
            start_epoch = self.load_checkpoint()

        print(f"\n{'='*50}")
        print(f"V7 ENHANCED TRAINING")
        print(f"{'='*50}")
        print(f"Target accuracy: {target_accuracy}%")
        print(f"V4 baseline: 8.75%")
        print(f"Experiment: {self.exp_dir}")
        print(f"Device: {self.device}")
        print(f"Epochs: {start_epoch+1} -> {self.num_epochs}")
        print(f"{'='*50}\n")

        for epoch in range(start_epoch + 1, self.num_epochs + 1):
            print(f"\n--- Epoch {epoch}/{self.num_epochs} ---")

            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader)

            # Gradient health check
            self.gradient_monitor.check_gradient_health()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_acc)
                print(f"  NEW BEST: {val_acc:.2f}%")

                if val_acc >= target_accuracy:
                    print(f"\n  TARGET {target_accuracy}% ACHIEVED!")
                    break
            else:
                self.patience_counter += 1

            # Save latest checkpoint every epoch (for resume)
            self.save_latest_checkpoint(epoch, val_acc)

            # Early stopping
            if self.patience_counter >= 15:
                print("\n  Early stopping triggered")
                break

            # Plot curves every 5 epochs
            if epoch % 5 == 0:
                self._plot_training_curves()

        self._print_final_summary(target_accuracy)
        return self.best_val_acc

    def _save_checkpoint(self, epoch, val_acc):
        """Save best model checkpoint"""
        import shutil
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_quantum': self.optimizer.quantum_optimizer.state_dict(),
            'optimizer_classical': self.optimizer.classical_optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }

        # Save to experiment dir and as latest best
        torch.save(checkpoint, os.path.join(self.exp_dir, 'best_model.pth'))
        torch.save(checkpoint, 'models/best_v7_model.pth')

        # Also backup best model to Drive
        if self.drive_backup_path:
            shutil.copy('models/best_v7_model.pth',
                        os.path.join(self.drive_backup_path, 'best_v7_model.pth'))

        # Save metadata
        metadata = {
            'experiment_name': self.experiment_name,
            'epoch': epoch,
            'val_accuracy': val_acc,
            'v4_baseline': 8.75,
            'improvement_over_v4': val_acc - 8.75,
        }
        with open(os.path.join(self.exp_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    def _plot_training_curves(self):
        """Generate training curve plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title('Loss')

        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train')
        axes[0, 1].plot(self.history['val_acc'], label='Val')
        axes[0, 1].axhline(y=8.75, color='r', linestyle='--', label='V4 Baseline (8.75%)')
        axes[0, 1].axhline(y=25, color='g', linestyle='--', label='V7 Target (25%)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].set_title('Accuracy')

        # Gradient norms
        qh = self.gradient_monitor.gradient_history
        if qh['quantum']:
            axes[1, 0].plot(qh['quantum'], label='Quantum', alpha=0.7)
            axes[1, 0].plot(qh['classical'], label='Classical', alpha=0.7)
            axes[1, 0].set_xlabel('Log Step')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
            axes[1, 0].set_title('Gradient Flow')

        # Improvement over V4
        improvements = [acc - 8.75 for acc in self.history['val_acc']]
        axes[1, 1].plot(improvements)
        axes[1, 1].axhline(y=16.25, color='g', linestyle='--', label='V7 Target (+16.25%)')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Improvement over V4 (%)')
        axes[1, 1].legend()
        axes[1, 1].set_title('Progress')

        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, 'training_curves.png'), dpi=100)
        plt.close()

    def _print_final_summary(self, target_accuracy):
        print(f"\n{'='*50}")
        print("TRAINING SUMMARY")
        print(f"{'='*50}")
        print(f"V4 Baseline Accuracy: 8.75%")
        print(f"Best Val Accuracy: {self.best_val_acc:.2f}%")
        print(f"Improvement over V4: {self.best_val_acc - 8.75:+.2f}%")
        print(f"Target ({target_accuracy}%): "
              f"{'ACHIEVED' if self.best_val_acc >= target_accuracy else 'NOT YET'}")
        print(f"Experiment: {self.exp_dir}")
        print(f"{'='*50}")


# -----------------
# Utility Functions
# -----------------

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_pred = torch.log_softmax(pred, dim=-1)
        nll_loss = -log_pred.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_pred.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def run_enhanced_training(circuit_type='data_reuploading', num_epochs=50,
                          resume=False, drive_backup_path=None):
    """
    Main entry point for V7 enhanced training.

    Args:
        circuit_type: 'strongly_entangling', 'data_reuploading', or 'hardware_efficient'
        num_epochs: Maximum training epochs
        resume: If True, resume from latest checkpoint
        drive_backup_path: If set, checkpoints are also saved to Drive (survives restarts)

    Returns:
        (best_val_acc, test_acc) tuple
    """
    device = torch.device(config.DEVICE)

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders()

    # Create V7 model
    model = create_enhanced_model(circuit_type=circuit_type, num_classes=config.NUM_CLASSES)

    # Create trainer
    trainer = EnhancedTrainer(
        model, train_loader, val_loader, device,
        num_epochs=num_epochs,
        experiment_name=f"v7_{circuit_type}",
        drive_backup_path=drive_backup_path
    )

    # Train (with optional resume)
    best_acc = trainer.train(train_loader, val_loader, target_accuracy=25.0, resume=resume)

    # Final test evaluation
    print("\nFinal evaluation on test set...")
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")

    return best_acc, test_acc
