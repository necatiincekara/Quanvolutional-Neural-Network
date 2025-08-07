"""
Improved training pipeline with quantum-aware optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import wandb
from collections import defaultdict


class QuantumAwareOptimizer:
    """
    Optimizer wrapper that handles quantum gradient peculiarities.
    """
    def __init__(self, model, base_lr=0.001, quantum_lr=0.0001):
        # Separate parameter groups for quantum and classical layers
        quantum_params = []
        classical_params = []
        
        for name, param in model.named_parameters():
            if 'quanv' in name or 'qlayer' in name:
                quantum_params.append(param)
            else:
                classical_params.append(param)
        
        # Use different learning rates and optimizers
        self.quantum_opt = torch.optim.Adam(quantum_params, lr=quantum_lr, 
                                           betas=(0.9, 0.999), weight_decay=1e-5)
        self.classical_opt = torch.optim.AdamW(classical_params, lr=base_lr, 
                                              weight_decay=1e-4)
        
        # Gradient clipping thresholds
        self.quantum_clip = 0.5
        self.classical_clip = 1.0
    
    def zero_grad(self):
        self.quantum_opt.zero_grad()
        self.classical_opt.zero_grad()
    
    def step(self):
        # Clip gradients before stepping
        torch.nn.utils.clip_grad_norm_(
            self.quantum_opt.param_groups[0]['params'], self.quantum_clip
        )
        torch.nn.utils.clip_grad_norm_(
            self.classical_opt.param_groups[0]['params'], self.classical_clip
        )
        
        self.quantum_opt.step()
        self.classical_opt.step()
    
    def state_dict(self):
        return {
            'quantum': self.quantum_opt.state_dict(),
            'classical': self.classical_opt.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.quantum_opt.load_state_dict(state_dict['quantum'])
        self.classical_opt.load_state_dict(state_dict['classical'])


class MixupAugmentation:
    """
    Mixup augmentation for better generalization.
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing for better generalization.
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        
        # One-hot encode target
        target_one_hot = torch.zeros_like(log_preds).scatter(1, target.unsqueeze(1), 1)
        
        # Apply label smoothing
        target_smooth = target_one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        loss = -(target_smooth * log_preds).sum(dim=-1).mean()
        return loss


def compute_gradient_statistics(model):
    """
    Compute gradient statistics for monitoring training health.
    """
    stats = defaultdict(list)
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            layer_type = 'quantum' if 'quanv' in name else 'classical'
            
            stats[f'{layer_type}_grad_mean'].append(grad.mean().item())
            stats[f'{layer_type}_grad_std'].append(grad.std().item())
            stats[f'{layer_type}_grad_norm'].append(grad.norm(2).item())
            
            # Check for vanishing/exploding gradients
            if grad.abs().mean() < 1e-7:
                stats['vanishing_gradients'].append(name)
            elif grad.abs().mean() > 10:
                stats['exploding_gradients'].append(name)
    
    return {k: np.mean(v) if k.endswith(('mean', 'std', 'norm')) else v 
            for k, v in stats.items()}


def train_epoch_improved(model, train_loader, optimizer, criterion, 
                        device, epoch, mixup=None, use_amp=True):
    """
    Improved training loop with better monitoring and augmentation.
    """
    model.train()
    
    scaler = GradScaler(enabled=use_amp)
    running_loss = 0.0
    correct = 0
    total = 0
    
    gradient_stats_accumulator = defaultdict(list)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        # Apply mixup if enabled
        if mixup and np.random.random() > 0.5:
            images, labels_a, labels_b, lam = mixup(images, labels)
            mixed = True
        else:
            mixed = False
        
        optimizer.zero_grad()
        
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            
            if mixed:
                loss = lam * criterion(outputs, labels_a) + \
                       (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        
        # Compute gradient statistics periodically
        if batch_idx % 10 == 0:
            grad_stats = compute_gradient_statistics(model)
            for k, v in grad_stats.items():
                if isinstance(v, (int, float)):
                    gradient_stats_accumulator[k].append(v)
        
        # Custom optimizer step or standard step
        if isinstance(optimizer, QuantumAwareOptimizer):
            scaler.unscale_(optimizer.quantum_opt)
            scaler.unscale_(optimizer.classical_opt)
            optimizer.step()
        else:
            scaler.step(optimizer)
        
        scaler.update()
        
        # Statistics
        running_loss += loss.item()
        if not mixed:
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': 100. * correct / total if total > 0 else 0
        })
    
    # Average gradient statistics
    avg_grad_stats = {k: np.mean(v) for k, v in gradient_stats_accumulator.items() 
                      if len(v) > 0}
    
    return running_loss / len(train_loader), 100. * correct / total, avg_grad_stats


def evaluate_with_uncertainty(model, val_loader, criterion, device, n_samples=5):
    """
    Evaluation with uncertainty estimation using Monte Carlo dropout.
    """
    model.eval()
    
    # Enable dropout for uncertainty estimation
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    
    all_losses = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            # Multiple forward passes for uncertainty
            batch_predictions = []
            for _ in range(n_samples):
                outputs = model(images)
                batch_predictions.append(F.softmax(outputs, dim=1))
            
            # Average predictions
            mean_pred = torch.stack(batch_predictions).mean(0)
            std_pred = torch.stack(batch_predictions).std(0)
            
            loss = criterion(torch.log(mean_pred + 1e-8), labels)
            all_losses.append(loss.item())
            
            # Store predictions and labels
            all_predictions.append(mean_pred.cpu())
            all_labels.append(labels.cpu())
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    _, predicted = all_predictions.max(1)
    accuracy = 100. * predicted.eq(all_labels).sum().item() / len(all_labels)
    avg_loss = np.mean(all_losses)
    
    # Compute per-class accuracy
    per_class_acc = []
    for c in range(all_predictions.size(1)):
        mask = all_labels == c
        if mask.sum() > 0:
            class_acc = predicted[mask].eq(all_labels[mask]).float().mean().item()
            per_class_acc.append(class_acc)
    
    return avg_loss, accuracy, per_class_acc


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


def train_quantum_model(model, train_loader, val_loader, config):
    """
    Main training function with all improvements.
    """
    device = torch.device(config.get('device', 'cuda'))
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = QuantumAwareOptimizer(
        model, 
        base_lr=config.get('base_lr', 0.001),
        quantum_lr=config.get('quantum_lr', 0.0001)
    )
    
    # Loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer.classical_opt, T_0=10, T_mult=2
    )
    
    # Augmentation
    mixup = MixupAugmentation(alpha=0.2)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15)
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(config.get('num_epochs', 50)):
        # Train
        train_loss, train_acc, grad_stats = train_epoch_improved(
            model, train_loader, optimizer, criterion, 
            device, epoch, mixup=mixup
        )
        
        # Evaluate
        val_loss, val_acc, per_class_acc = evaluate_with_uncertainty(
            model, val_loader, criterion, device
        )
        
        # Learning rate scheduling
        scheduler.step()
        
        # Logging
        print(f"Epoch {epoch+1}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Gradient Stats: {grad_stats}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, 'best_quantum_model.pth')
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return model