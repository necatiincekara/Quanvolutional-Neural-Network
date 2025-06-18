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
    
    # Single trainable layer of Rot gates followed by nearest-neighbour entanglement
    for i in range(config.N_QUBITS):
        qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=i)

    # Nearest-neighbour CNOT chain (i -> i+1)
    for i in range(config.N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    
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
        # Updated weight shape for the simplified circuit: (n_qubits, 3)
        weight_shapes = {"weights": (n_qubits, 3)}
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

        # Ensure the output is on the same device as the input
        return processed_patches.to(x.device)

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
        # 1) Classical pre-processing: Conv(1→4) + ReLU + 2×2 MaxPool ⇒ 32×32 → 16×16, channels 4
        self.pre = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 2) Quantum layer (operates on 4-channel 16×16 feature map)
        self.quanv = QuanvLayer(n_qubits=n_qubits)

        # 3) Deeper classical processing after quantum layer
        in_channels_after_quanv = 4 * n_qubits  # Quanv merges channel & qubit dims
        self.conv1 = nn.Conv2d(in_channels_after_quanv, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)  # 16×16 → 8×8 after Quanv; pool→4×4

        # Determine flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
            dummy = self.pre(dummy)
            dummy = self.quanv(dummy)
            dummy = self.pool(torch.relu(self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(dummy)))))))
            flatten_dim = dummy.numel()

        self.fc1 = nn.Linear(flatten_dim, config.FC1_OUTPUT)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(config.FC1_OUTPUT, num_classes)

    def forward(self, x):
        # 1) Classical pre-processing
        x = self.pre(x)

        # 2) Quantum feature extraction
        x = self.quanv(x)

        # 3) Classical convolutional stack
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Flatten and classify
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x) # Apply dropout before the final layer
        x = self.fc2(x)
        return x 