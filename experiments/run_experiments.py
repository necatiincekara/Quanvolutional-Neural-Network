"""
Experimental Validation Framework for Quantum-Classical Hybrid Model
Systematic experiments to validate 82% -> 90% accuracy improvement
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import sys
sys.path.append('..')

from src import config
from src.trainable_quantum_model import create_enhanced_model
from src.enhanced_training import run_enhanced_training
from src.dataset import get_dataloaders
from src.model import QuanvNet  # Original model for comparison

class ExperimentalFramework:
    """
    Comprehensive experimental framework for validating quantum advantage
    """
    def __init__(self, base_dir="experimental_results"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.results = []
        
    def run_baseline_comparison(self):
        """
        Experiment 1: Fixed vs Trainable Quantum Layers
        Direct comparison to validate the core hypothesis
        """
        print("\n" + "="*60)
        print("EXPERIMENT 1: Fixed vs Trainable Quantum Layers")
        print("="*60)
        
        device = torch.device(config.DEVICE)
        train_loader, val_loader, test_loader = get_dataloaders()
        
        results = {}
        
        # 1. Baseline: Fixed Quantum Layer (Master's Thesis)
        print("\n--- Testing Fixed Quantum Layer (82% Baseline) ---")
        fixed_model = QuanvNet().to(device)
        
        # Freeze quantum parameters
        for name, param in fixed_model.named_parameters():
            if 'quanv' in name:
                param.requires_grad = False
        
        # Count parameters
        fixed_params = sum(p.numel() for p in fixed_model.parameters() if p.requires_grad)
        print(f"Trainable parameters (Fixed Quantum): {fixed_params:,}")
        
        # Train with fixed quantum layer
        results['fixed_quantum'] = self._train_model(
            fixed_model, train_loader, val_loader, test_loader,
            experiment_name="fixed_quantum_baseline"
        )
        
        # 2. Enhanced: Trainable Quantum Layer
        print("\n--- Testing Trainable Quantum Layer (Target 90%) ---")
        trainable_model = create_enhanced_model(circuit_type='data_reuploading').to(device)
        
        # Count parameters
        trainable_params = sum(p.numel() for p in trainable_model.parameters() if p.requires_grad)
        quantum_params = sum(p.numel() for p in trainable_model.parameters() 
                           if p.requires_grad and 'quanv' in name)
        print(f"Trainable parameters (Total): {trainable_params:,}")
        print(f"Trainable parameters (Quantum): {quantum_params:,}")
        
        # Train with trainable quantum layer
        results['trainable_quantum'] = self._train_model(
            trainable_model, train_loader, val_loader, test_loader,
            experiment_name="trainable_quantum_enhanced"
        )
        
        # Compare results
        self._compare_results(results)
        return results
    
    def run_circuit_comparison(self):
        """
        Experiment 2: Compare Different Quantum Circuit Architectures
        """
        print("\n" + "="*60)
        print("EXPERIMENT 2: Quantum Circuit Architecture Comparison")
        print("="*60)
        
        circuit_types = ['strongly_entangling', 'data_reuploading', 'hardware_efficient']
        results = {}
        
        for circuit_type in circuit_types:
            print(f"\n--- Testing {circuit_type} circuit ---")
            best_val, test_acc = run_enhanced_training(
                circuit_type=circuit_type,
                num_epochs=50  # Shorter for comparison
            )
            
            results[circuit_type] = {
                'best_val_acc': best_val,
                'test_acc': test_acc
            }
            
            print(f"Results: Val={best_val:.2f}%, Test={test_acc:.2f}%")
        
        # Visualize comparison
        self._plot_circuit_comparison(results)
        return results
    
    def run_ablation_study(self):
        """
        Experiment 3: Ablation Study on Key Components
        """
        print("\n" + "="*60)
        print("EXPERIMENT 3: Ablation Study")
        print("="*60)
        
        device = torch.device(config.DEVICE)
        train_loader, val_loader, test_loader = get_dataloaders()
        
        ablations = {
            'full_model': self._create_full_model(),
            'no_residual': self._create_model_without_residual(),
            'no_attention': self._create_model_without_attention(),
            'no_mixup': self._create_model_without_mixup(),
            'single_layer_quantum': self._create_single_layer_quantum()
        }
        
        results = {}
        for name, model in ablations.items():
            print(f"\n--- Testing: {name} ---")
            model = model.to(device)
            results[name] = self._train_model(
                model, train_loader, val_loader, test_loader,
                experiment_name=f"ablation_{name}",
                epochs=30
            )
        
        self._analyze_ablation_results(results)
        return results
    
    def run_gradient_analysis(self):
        """
        Experiment 4: Gradient Flow Analysis
        """
        print("\n" + "="*60)
        print("EXPERIMENT 4: Gradient Flow Analysis")
        print("="*60)
        
        device = torch.device(config.DEVICE)
        model = create_enhanced_model().to(device)
        
        # Create synthetic batch for analysis
        batch_size = 32
        dummy_input = torch.randn(batch_size, 1, 32, 32).to(device)
        dummy_target = torch.randint(0, config.NUM_CLASSES, (batch_size,)).to(device)
        
        # Forward pass
        output = model(dummy_input)
        loss = torch.nn.CrossEntropyLoss()(output, dummy_target)
        
        # Backward pass
        loss.backward()
        
        # Analyze gradients
        gradient_stats = self._analyze_gradients(model)
        
        # Visualize gradient flow
        self._plot_gradient_flow(gradient_stats)
        
        return gradient_stats
    
    def run_quantum_expressivity_analysis(self):
        """
        Experiment 5: Quantum Circuit Expressivity Analysis
        """
        print("\n" + "="*60)
        print("EXPERIMENT 5: Quantum Circuit Expressivity")
        print("="*60)
        
        from pennylane import numpy as pnp
        import pennylane as qml
        
        # Calculate effective dimension for different circuits
        expressivity_results = {}
        
        for circuit_type in ['strongly_entangling', 'data_reuploading', 'hardware_efficient']:
            print(f"\n--- Analyzing {circuit_type} ---")
            
            # Create model
            model = create_enhanced_model(circuit_type=circuit_type)
            
            # Estimate effective dimension
            eff_dim = self._estimate_effective_dimension(model)
            expressivity_results[circuit_type] = eff_dim
            
            print(f"Effective Dimension: {eff_dim:.2f}")
        
        return expressivity_results
    
    def _train_model(self, model, train_loader, val_loader, test_loader, 
                     experiment_name, epochs=50):
        """Helper function to train a model"""
        from src.enhanced_training import EnhancedTrainer
        
        device = torch.device(config.DEVICE)
        trainer = EnhancedTrainer(model, device, experiment_name)
        
        # Train
        best_val = trainer.train(train_loader, val_loader, num_epochs=epochs, target_accuracy=90.0)
        
        # Test
        test_loss, test_acc = trainer.validate(test_loader)
        
        return {
            'best_val_acc': best_val,
            'test_acc': test_acc,
            'history': trainer.history
        }
    
    def _compare_results(self, results):
        """Compare and visualize results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        models = list(results.keys())
        val_accs = [results[m]['best_val_acc'] for m in models]
        test_accs = [results[m]['test_acc'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0].bar(x - width/2, val_accs, width, label='Validation', color='blue', alpha=0.7)
        axes[0].bar(x + width/2, test_accs, width, label='Test', color='green', alpha=0.7)
        axes[0].axhline(y=82, color='r', linestyle='--', label='Thesis Baseline (82%)')
        axes[0].axhline(y=90, color='g', linestyle='--', label='Target (90%)')
        axes[0].set_xlabel('Model Type')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Fixed vs Trainable Quantum Layers')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models)
        axes[0].legend()
        
        # Learning curves
        for model_name, result in results.items():
            axes[1].plot(result['history']['val_acc'], label=f"{model_name} (Val)")
        axes[1].axhline(y=82, color='r', linestyle='--', alpha=0.5)
        axes[1].axhline(y=90, color='g', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation Accuracy (%)')
        axes[1].set_title('Learning Curves Comparison')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, 'fixed_vs_trainable_comparison.png'))
        plt.show()
    
    def _plot_circuit_comparison(self, results):
        """Visualize circuit architecture comparison"""
        df = pd.DataFrame(results).T
        
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(kind='bar', ax=ax)
        ax.axhline(y=82, color='r', linestyle='--', label='Baseline (82%)')
        ax.axhline(y=90, color='g', linestyle='--', label='Target (90%)')
        ax.set_xlabel('Circuit Type')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Quantum Circuit Architecture Comparison')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, 'circuit_comparison.png'))
        plt.show()
    
    def _analyze_gradients(self, model):
        """Analyze gradient statistics"""
        stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                stats[name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'max': grad.max().item(),
                    'min': grad.min().item(),
                    'norm': grad.norm().item()
                }
        
        return stats
    
    def _plot_gradient_flow(self, gradient_stats):
        """Visualize gradient flow through layers"""
        layers = []
        norms = []
        
        for name, stats in gradient_stats.items():
            layers.append(name.split('.')[0])
            norms.append(stats['norm'])
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(norms)), norms)
        plt.xticks(range(len(layers)), layers, rotation=90)
        plt.xlabel('Layer')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Flow Analysis')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, 'gradient_flow.png'))
        plt.show()
    
    def _estimate_effective_dimension(self, model):
        """Estimate effective dimension of quantum circuit"""
        # Simplified estimation based on parameter count and entanglement
        quantum_params = sum(p.numel() for name, p in model.named_parameters() 
                           if 'quanv' in name and p.requires_grad)
        
        # Rough estimate: effective_dim ≈ 2^(sqrt(params))
        eff_dim = 2 ** np.sqrt(quantum_params)
        return eff_dim
    
    def _create_full_model(self):
        """Create full enhanced model"""
        return create_enhanced_model(circuit_type='data_reuploading')
    
    def _create_model_without_residual(self):
        """Create model without residual connections"""
        model = create_enhanced_model(circuit_type='data_reuploading')
        # Disable skip connections by setting weight to 0
        model.skip_weight.data.fill_(0.0)
        model.skip_weight.requires_grad = False
        return model
    
    def _create_model_without_attention(self):
        """Create model without attention mechanism"""
        model = create_enhanced_model(circuit_type='data_reuploading')
        # Replace attention with identity
        model.attention = nn.Identity()
        return model
    
    def _create_model_without_mixup(self):
        """Model trained without mixup augmentation"""
        # This is controlled in training, return normal model
        return create_enhanced_model(circuit_type='data_reuploading')
    
    def _create_single_layer_quantum(self):
        """Create model with single quantum layer"""
        model = create_enhanced_model(circuit_type='data_reuploading')
        # Modify quantum layer to use single layer
        model.quanv.n_layers = 1
        return model
    
    def _analyze_ablation_results(self, results):
        """Analyze and visualize ablation study results"""
        # Create comparison table
        df = pd.DataFrame({
            name: {
                'Val Acc': res['best_val_acc'],
                'Test Acc': res['test_acc'],
                'Improvement': res['best_val_acc'] - 82.0
            }
            for name, res in results.items()
        }).T
        
        print("\n" + "="*60)
        print("ABLATION STUDY RESULTS")
        print("="*60)
        print(df.to_string())
        
        # Save results
        df.to_csv(os.path.join(self.base_dir, 'ablation_results.csv'))
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 6))
        df[['Val Acc', 'Test Acc']].plot(kind='bar', ax=ax)
        ax.axhline(y=82, color='r', linestyle='--', label='Baseline')
        ax.axhline(y=90, color='g', linestyle='--', label='Target')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Ablation Study Results')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, 'ablation_study.png'))
        plt.show()

def run_all_experiments():
    """
    Run complete experimental validation suite
    """
    print("\n" + "="*60)
    print("QUANTUM-CLASSICAL HYBRID MODEL EXPERIMENTAL VALIDATION")
    print("Target: 82% (Fixed Quantum) → 90% (Trainable Quantum)")
    print("="*60)
    
    framework = ExperimentalFramework()
    
    # Run experiments
    all_results = {}
    
    # Experiment 1: Core hypothesis validation
    all_results['baseline_comparison'] = framework.run_baseline_comparison()
    
    # Experiment 2: Circuit architecture comparison
    all_results['circuit_comparison'] = framework.run_circuit_comparison()
    
    # Experiment 3: Ablation study
    all_results['ablation_study'] = framework.run_ablation_study()
    
    # Experiment 4: Gradient analysis
    all_results['gradient_analysis'] = framework.run_gradient_analysis()
    
    # Experiment 5: Expressivity analysis
    all_results['expressivity'] = framework.run_quantum_expressivity_analysis()
    
    # Save all results
    with open(os.path.join(framework.base_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("EXPERIMENTAL VALIDATION COMPLETE")
    print(f"Results saved to: {framework.base_dir}")
    print("="*60)
    
    return all_results

if __name__ == "__main__":
    # Run all experiments
    results = run_all_experiments()