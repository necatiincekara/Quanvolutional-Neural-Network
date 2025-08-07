"""
Enhanced Trainable Quantum-Classical Hybrid Model for 90% Accuracy Target
Building upon the 82% baseline with fixed quantum layers from master's thesis
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from . import config

# Enable quantum compilation cache
qml.transforms.dynamic_dispatch.enable_tape_cache()

# -----------------
# Enhanced Quantum Circuits with Multiple Parameterization Strategies
# -----------------

def create_quantum_device(n_qubits=4):
    """Create quantum device with optimal settings for training"""
    return qml.device(
        config.QUANTUM_DEVICE,
        wires=n_qubits,
        shots=None,  # Use exact expectation values for training stability
        batch_obs=True  # Enable batched observations
    )

# Strategy 1: Strongly Entangling Layers (Best for gradient flow)
@qml.qnode(create_quantum_device(), interface='torch', diff_method='adjoint')
def strongly_entangling_circuit(inputs, weights):
    """
    Strongly entangling circuit with multiple layers for expressivity
    Achieves better gradient flow than simple circuits
    """
    n_qubits = 4
    n_layers = 3  # Optimal depth based on literature
    
    # Input encoding with angle embedding
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # Strongly entangling layers
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Return expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Strategy 2: Data Re-uploading Circuit (Best for accuracy)
@qml.qnode(create_quantum_device(), interface='torch', diff_method='adjoint')
def data_reuploading_circuit(inputs, weights):
    """
    Data re-uploading strategy for enhanced expressivity
    Re-encodes data at each layer to increase model capacity
    """
    n_qubits = 4
    n_layers = 2
    
    for layer in range(n_layers):
        # Re-upload data at each layer
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        
        # Parameterized rotation layer
        for i in range(n_qubits):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        
        # Entangling layer with circular connectivity
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
        
        # Additional rotation layer for expressivity
        for i in range(n_qubits):
            qml.RY(weights[layer, i, 2], wires=i)
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Strategy 3: Hardware Efficient Ansatz (Best for NISQ devices)
@qml.qnode(create_quantum_device(), interface='torch', diff_method='adjoint')
def hardware_efficient_circuit(inputs, weights):
    """
    Hardware-efficient ansatz optimized for near-term quantum devices
    Balances expressivity with circuit depth
    """
    n_qubits = 4
    n_layers = 2
    
    # Amplitude encoding for better information density
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
    
    for layer in range(n_layers):
        # Single-qubit rotations
        for i in range(n_qubits):
            qml.RX(weights[layer, i, 0], wires=i)
            qml.RY(weights[layer, i, 1], wires=i)
        
        # Entangling gates with alternating patterns
        if layer % 2 == 0:
            for i in range(0, n_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])
        else:
            for i in range(1, n_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])
            qml.CZ(wires=[n_qubits - 1, 0])  # Circular boundary
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# -----------------
# Enhanced Trainable Quantum Layer
# -----------------

class TrainableQuanvLayer(nn.Module):
    """
    Fully trainable quantum convolutional layer with multiple circuit options
    Implements gradient stabilization and parameter initialization strategies
    """
    def __init__(self, n_qubits=4, circuit_type='data_reuploading', n_layers=2):
        super(TrainableQuanvLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit_type = circuit_type
        
        # Select circuit and define weight shapes
        if circuit_type == 'strongly_entangling':
            self.circuit = strongly_entangling_circuit
            # StronglyEntanglingLayers expects (n_layers, n_qubits, 3)
            weight_shapes = {"weights": (3, n_qubits, 3)}
        elif circuit_type == 'data_reuploading':
            self.circuit = data_reuploading_circuit
            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        elif circuit_type == 'hardware_efficient':
            self.circuit = hardware_efficient_circuit
            weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
        
        # Create quantum layer
        self.qlayer = qml.qnn.TorchLayer(self.circuit, weight_shapes)
        
        # Initialize with variance-preserving strategy
        self._initialize_quantum_weights()
        
        # Gradient scaling factor (learned)
        self.gradient_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def _initialize_quantum_weights(self):
        """
        Initialize quantum weights using variance-preserving strategy
        Critical for avoiding barren plateaus
        """
        with torch.no_grad():
            for name, param in self.qlayer.named_parameters():
                if 'weight' in name:
                    # Use smaller initialization for quantum parameters
                    nn.init.normal_(param, mean=0.0, std=0.01)
    
    def forward(self, x):
        """
        Forward pass with gradient scaling for stability
        """
        # Extract patches (same as before)
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
        batch_size, channels, out_h, out_w, _, _ = patches.size()
        
        # Reshape for quantum processing
        patches = patches.reshape(batch_size, channels, out_h * out_w, -1)
        patches = patches.permute(0, 2, 1, 3).reshape(-1, self.n_qubits)
        
        # Process through quantum circuit with gradient scaling
        processed_patches = self.qlayer(patches) * self.gradient_scale
        
        # Reshape back
        final_shape = (batch_size, out_h * out_w, channels, self.n_qubits)
        processed_patches = processed_patches.view(final_shape)
        processed_patches = processed_patches.permute(0, 2, 3, 1)
        processed_patches = processed_patches.reshape(batch_size, channels * self.n_qubits, out_h, out_w)
        
        return processed_patches.to(x.device)

# -----------------
# Enhanced Hybrid Model for 90% Accuracy Target
# -----------------

class EnhancedQuanvNet(nn.Module):
    """
    Enhanced quantum-classical hybrid model targeting 90% accuracy
    Improvements over 82% baseline:
    1. Trainable quantum parameters (vs fixed)
    2. Residual connections for gradient flow
    3. Attention mechanisms for feature selection
    4. Multi-scale processing
    """
    def __init__(self, n_qubits=4, num_classes=44, circuit_type='data_reuploading'):
        super(EnhancedQuanvNet, self).__init__()
        
        # Classical preprocessing with residual blocks
        self.pre = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # 32->16
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            ResidualBlock(8, 8),
            nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),  # 16->8
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        
        # Trainable quantum layer (key difference from 82% baseline)
        self.quanv = TrainableQuanvLayer(
            n_qubits=n_qubits,
            circuit_type=circuit_type,
            n_layers=2
        )
        
        # Attention gate for quantum features
        self.attention = SelfAttention(channels=16)  # 4 * n_qubits
        
        # Enhanced classical processing
        self.post = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2)
        )
        
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Skip connection weight (learnable)
        self.skip_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        # Preprocessing
        pre_features = self.pre(x)
        
        # Quantum processing with skip connection
        quantum_features = self.quanv(pre_features)
        
        # Apply attention to quantum features
        quantum_features = self.attention(quantum_features)
        
        # Residual connection if shapes match
        if quantum_features.shape[1] == pre_features.shape[1] * 4:
            # Expand pre_features to match quantum_features channels
            pre_expanded = pre_features.repeat(1, 4, 1, 1)
            quantum_features = quantum_features + self.skip_weight * pre_expanded
        
        # Post-processing
        features = self.post(quantum_features)
        
        # Classification
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        
        return output

# -----------------
# Supporting Modules
# -----------------

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)

class SelfAttention(nn.Module):
    """Self-attention mechanism for quantum feature selection"""
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Compute attention
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        
        proj_value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Apply attention with learnable weight
        out = self.gamma * out + x
        return out

# -----------------
# Model Factory
# -----------------

def create_enhanced_model(circuit_type='data_reuploading', num_classes=44):
    """
    Factory function to create enhanced model with specified circuit type
    
    Args:
        circuit_type: 'strongly_entangling', 'data_reuploading', or 'hardware_efficient'
        num_classes: Number of output classes
    
    Returns:
        Enhanced quantum-classical hybrid model
    """
    return EnhancedQuanvNet(
        n_qubits=4,
        num_classes=num_classes,
        circuit_type=circuit_type
    )