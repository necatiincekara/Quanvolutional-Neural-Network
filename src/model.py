"""
Quantum and classical model definitions for the Quanvolutional Neural Network.
"""

import torch
import torch.nn as nn
import pennylane as qml
from . import config

# -----------------
# Quantum Components
# -----------------

# Define the quantum device
dev = qml.device(config.QUANTUM_DEVICE, wires=config.N_QUBITS)

@qml.qnode(dev, interface='torch', diff_method='adjoint')
def quanv_circuit(inputs, weights):
    """
    A more expressive quantum circuit for the quanvolutional layer.
    It uses angle embedding for inputs and layers of Rotations and CNOTs.
    This structure allows the model to learn more complex functions.
    """
    # Encode input data from the image patch
    qml.AngleEmbedding(inputs, wires=range(config.N_QUBITS))
    
    # Add trainable layers
    for layer_weights in weights:
        # Layer of rotation gates
        for i in range(config.N_QUBITS):
            qml.Rot(layer_weights[i, 0], layer_weights[i, 1], layer_weights[i, 2], wires=i)
        
        # Entangling layer (circular CNOTs)
        for i in range(config.N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % config.N_QUBITS])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(config.N_QUBITS)]

# -----------------
# Hybrid Model Layers
# -----------------

class QuanvLayer(nn.Module):
    """
    The Quanvolutional Layer that applies the quantum circuit to image patches.
    This version is vectorized to leverage GPU parallelism.
    """
    def __init__(self, n_qubits):
        super(QuanvLayer, self).__init__()
        self.n_qubits = n_qubits
        # Updated weight shapes for the new circuit: (2 layers, n_qubits, 3 params per Rot gate)
        weight_shapes = {"weights": (2, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quanv_circuit, weight_shapes)

    def forward(self, x):
        # The vectorized forward pass logic from the previous step is assumed to be here
        # and remains unchanged. This is a placeholder for brevity.
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
        batch_size, channels, out_h, out_w, _, _ = patches.size()
        
        patches = patches.reshape(batch_size, channels, out_h * out_w, -1)
        patches = patches.permute(0, 2, 1, 3).reshape(-1, self.n_qubits)
        
        processed_patches = self.qlayer(patches)
        
        final_shape = (batch_size, out_h * out_w, channels, self.n_qubits)
        processed_patches = processed_patches.view(final_shape)
        
        processed_patches = processed_patches.permute(0, 2, 3, 1) 
        
        processed_patches = processed_patches.reshape(batch_size, channels * self.n_qubits, out_h, out_w)
        
        return processed_patches

# -----------------
# Full Hybrid Model
# -----------------

class QuanvNet(nn.Module):
    """
    The main Quanvolutional Neural Network model, combining the
    quantum layer with classical convolutional and fully connected layers.
    Includes Batch Normalization and Dropout for improved training.
    """
    def __init__(self, n_qubits=config.N_QUBITS, num_classes=config.NUM_CLASSES):
        super(QuanvNet, self).__init__()
        self.quanv = QuanvLayer(n_qubits=n_qubits)
        # After the QuanvLayer, we have `channels * n_qubits` feature maps.
        # Assuming input has 1 channel from our dataset loader.
        self.conv = nn.Conv2d(1 * n_qubits, 16, kernel_size=config.CONV_KERNEL_SIZE)
        self.bn1 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(config.FC1_INPUT, config.FC1_OUTPUT)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(config.FC1_OUTPUT, num_classes)

    def forward(self, x):
        x = self.quanv(x) 
        x = torch.relu(self.bn1(self.conv(x)))
        x = x.reshape(-1, config.FC1_INPUT) # Flatten the tensor for the fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.dropout(x) # Apply dropout before the final layer
        x = self.fc2(x)
        return x 