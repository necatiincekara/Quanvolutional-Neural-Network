"""
Enhanced Trainable Quantum-Classical Hybrid Model (V7)
Gradient-stabilized architecture targeting 25%+ accuracy
Building upon V4 stable baseline (8x8 feature maps, 8.75% accuracy)
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from . import config

# Enable quantum compilation cache
try:
    qml.transforms.dynamic_dispatch.enable_tape_cache()
except AttributeError:
    pass

# -----------------
# Quantum Device Factory
# -----------------

def create_quantum_device(n_qubits=4):
    """Create quantum device with optimal settings for training"""
    return qml.device(
        config.QUANTUM_DEVICE,
        wires=n_qubits,
    )

# -----------------
# Quantum Circuit Strategies
# -----------------

# Strategy 1: Strongly Entangling Layers (Best for gradient flow)
@qml.qnode(create_quantum_device(), interface='torch', diff_method='adjoint')
def strongly_entangling_circuit(inputs, weights):
    """
    Strongly entangling circuit with multiple layers for expressivity.
    Achieves better gradient flow than simple circuits.
    """
    n_qubits = 4

    # Input encoding with angle embedding
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

    # Strongly entangling layers
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Strategy 2: Data Re-uploading Circuit (Best for accuracy)
@qml.qnode(create_quantum_device(), interface='torch', diff_method='adjoint')
def data_reuploading_circuit(inputs, weights):
    """
    Data re-uploading strategy for enhanced expressivity.
    Re-encodes data at each layer to increase model capacity.
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
    Hardware-efficient ansatz optimized for near-term quantum devices.
    Uses AngleEmbedding (4 inputs for 4 qubits) instead of AmplitudeEmbedding
    which would require 2^4=16 inputs.
    """
    n_qubits = 4
    n_layers = 2

    # Angle encoding (matches 4-value patch input)
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

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
    Fully trainable quantum convolutional layer with gradient stabilization.
    Key improvements over base QuanvLayer:
    - Multiple circuit strategies
    - Learnable gradient scaling
    - Variance-preserving initialization (anti-barren plateau)
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

        # Learnable gradient scaling factor for stability
        self.gradient_scale = nn.Parameter(torch.ones(1) * 0.1)

    def _initialize_quantum_weights(self):
        """
        Initialize quantum weights to avoid barren plateaus.
        Uses small uniform initialization around zero.
        """
        with torch.no_grad():
            for name, param in self.qlayer.named_parameters():
                if 'weight' in name:
                    # Small random initialization - not too small to avoid barren plateau
                    nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, x):
        # Extract 2x2 patches with stride 2
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
        batch_size, channels, out_h, out_w, _, _ = patches.size()

        # Reshape: (batch*patches, n_qubits)
        patches = patches.reshape(batch_size, channels, out_h * out_w, -1)
        patches = patches.permute(0, 2, 1, 3).reshape(-1, self.n_qubits)

        # Quantum processing with gradient scaling
        processed_patches = self.qlayer(patches) * self.gradient_scale

        # Reshape back: (batch, channels*n_qubits, out_h, out_w)
        final_shape = (batch_size, out_h * out_w, channels, self.n_qubits)
        processed_patches = processed_patches.view(final_shape)
        processed_patches = processed_patches.permute(0, 2, 3, 1)
        processed_patches = processed_patches.reshape(
            batch_size, channels * self.n_qubits, out_h, out_w
        )

        return processed_patches.to(x.device)

# -----------------
# V7: Gradient-Stabilized Hybrid Model
# -----------------

class EnhancedQuanvNet(nn.Module):
    """
    V7 Enhanced quantum-classical hybrid model.

    Key improvements over V4 (8.75% accuracy):
    1. Trainable quantum parameters (vs fixed in V4)
    2. Residual connections around quantum layer for gradient flow
    3. Channel attention for quantum feature selection
    4. GroupNorm throughout (stable with small batch sizes)
    5. Learnable skip connection weight

    Architecture: 32x32 -> 16x16 -> 8x8 -> [Quantum 4x4] -> CNN -> 44 classes
    Feature map: 8x8 (proven optimal in V4, avoids V6 gradient collapse)
    """
    def __init__(self, n_qubits=4, num_classes=44, circuit_type='data_reuploading'):
        super(EnhancedQuanvNet, self).__init__()

        # Classical preprocessing: 32x32 -> 8x8 with 4 channels
        # Uses GroupNorm for stability with small effective batch sizes
        self.pre = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # 32->16
            nn.GroupNorm(4, 8),
            nn.GELU(),
            ResidualBlock(8, 8),
            nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),   # 16->8
            nn.GroupNorm(2, 4),
            nn.GELU()
        )

        # Trainable quantum layer on 8x8 feature map -> 4x4 output
        # 16 quantum circuit evaluations per image (same as V4)
        self.quanv = TrainableQuanvLayer(
            n_qubits=n_qubits,
            circuit_type=circuit_type,
            n_layers=2
        )

        # Quantum output channels: 4 channels * 4 qubits = 16
        quanv_out_channels = 4 * n_qubits

        # Channel attention for quantum features
        self.attention = ChannelAttention(quanv_out_channels)

        # Post-quantum classical processing
        self.post = nn.Sequential(
            nn.Conv2d(quanv_out_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(2)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        # Learnable skip connection weight (starts small)
        self.skip_weight = nn.Parameter(torch.tensor(0.1))

        # Adapter to match pre_features (4ch) to quantum output (16ch) for residual
        self.skip_adapter = nn.Conv2d(4, quanv_out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # 1) Classical preprocessing: (B,1,32,32) -> (B,4,8,8)
        pre_features = self.pre(x)

        # 2) Quantum processing: (B,4,8,8) -> (B,16,4,4)
        quantum_features = self.quanv(pre_features)

        # 3) Channel attention
        quantum_features = self.attention(quantum_features)

        # 4) Residual connection: adapt pre_features spatially and channel-wise
        #    pre_features is (B,4,8,8), quantum_features is (B,16,4,4)
        #    Downsample pre_features to match quantum spatial dims
        pre_downsampled = nn.functional.adaptive_avg_pool2d(
            pre_features, quantum_features.shape[2:]
        )
        pre_adapted = self.skip_adapter(pre_downsampled)
        quantum_features = quantum_features + self.skip_weight * pre_adapted

        # 5) Classical post-processing
        features = self.post(quantum_features)

        # 6) Classification
        features = features.reshape(features.size(0), -1)
        output = self.classifier(features)

        return output

# -----------------
# Supporting Modules
# -----------------

class ResidualBlock(nn.Module):
    """Residual block with GroupNorm for gradient flow stability"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(min(4, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(min(4, out_channels), out_channels)

    def forward(self, x):
        residual = x
        out = nn.functional.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + residual
        return nn.functional.gelu(out)


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    More stable than spatial self-attention for small feature maps.
    """
    def __init__(self, channels, reduction=4):
        super(ChannelAttention, self).__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.squeeze(x).view(b, c)
        w = self.excite(w).view(b, c, 1, 1)
        return x * w

# -----------------
# Model Factory
# -----------------

def create_enhanced_model(circuit_type='data_reuploading', num_classes=44):
    """
    Factory function to create V7 enhanced model.

    Args:
        circuit_type: 'strongly_entangling', 'data_reuploading', or 'hardware_efficient'
        num_classes: Number of output classes (44 for Ottoman characters)

    Returns:
        EnhancedQuanvNet model instance
    """
    return EnhancedQuanvNet(
        n_qubits=4,
        num_classes=num_classes,
        circuit_type=circuit_type
    )
