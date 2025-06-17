"""
Quantum and classical model definitions for the Quanvolutional Neural Network.
"""

import torch
import torch.nn as nn
import pennylane as qml
from src import config

# -----------------
# Quantum Components
# -----------------

# Define the quantum device
dev = qml.device(config.QUANTUM_DEVICE, wires=config.N_QUBITS)

@qml.qnode(dev, interface='torch', diff_method='backprop')
def quanv_circuit(inputs, weights):
    """
    The quantum circuit for the quanvolutional layer.
    It applies a layer of RY gates parameterized by input data,
    followed by a layer of entangling gates with trainable weights.
    """
    for i in range(config.N_QUBITS):
        qml.RY(inputs[i], wires=i)
    
    qml.templates.layers.BasicEntanglerLayers(weights, wires=range(config.N_QUBITS))
    
    return [qml.expval(qml.PauliZ(i)) for i in range(config.N_QUBITS)]

# -----------------
# Hybrid Model Layers
# -----------------

class QuanvLayer(nn.Module):
    """
    The Quanvolutional Layer that applies the quantum circuit to image patches.
    """
    def __init__(self, n_qubits):
        super(QuanvLayer, self).__init__()
        self.n_qubits = n_qubits
        weight_shapes = {"weights": (1, self.n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(quanv_circuit, weight_shapes)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        # The quantum circuit expects a 2x2 patch, which has 4 features.
        # This matches n_qubits=4.
        kernel_size = 2
        
        out_components = []
        for b in range(batch_size):
            # Process each image in the batch individually
            x_image = x[b]
            
            # Apply the quantum circuit to each 2x2 patch of the image
            # Unfold extracts sliding local blocks from a batched input tensor.
            patches = x_image.unfold(1, kernel_size, kernel_size).unfold(2, kernel_size, kernel_size)
            patches = patches.contiguous().view(channels, -1, kernel_size, kernel_size)
            
            q_results = []
            # Iterate over patches
            for c in range(patches.shape[0]):
                for p in range(patches.shape[1]):
                    patch = patches[c, p, :, :].flatten()
                    q_out = self.qlayer(patch)
                    q_results.append(q_out)
            
            # Reshape the quantum results into a feature map
            out_image = torch.stack(q_results).view(channels, height // kernel_size, width // kernel_size, self.n_qubits)
            out_image = out_image.permute(0, 3, 1, 2).contiguous()
            out_components.append(out_image)
            
        final_out = torch.stack(out_components)
        
        # The output of qlayer is (n_qubits), so we create n_qubits channels.
        # We merge the original channel dim with the qubit dim.
        final_out = final_out.view(batch_size, -1, height // kernel_size, width // kernel_size)
        return final_out

# -----------------
# Full Hybrid Model
# -----------------

class QuanvNet(nn.Module):
    """
    The main Quanvolutional Neural Network model, combining the
    quantum layer with classical convolutional and fully connected layers.
    """
    def __init__(self, n_qubits=config.N_QUBITS, num_classes=config.NUM_CLASSES):
        super(QuanvNet, self).__init__()
        self.quanv = QuanvLayer(n_qubits=n_qubits)
        # After the QuanvLayer, we have `n_qubits` feature maps (channels).
        self.conv = nn.Conv2d(n_qubits, 16, kernel_size=config.CONV_KERNEL_SIZE)
        self.fc1 = nn.Linear(config.FC1_INPUT, config.FC1_OUTPUT)
        self.fc2 = nn.Linear(config.FC1_OUTPUT, num_classes)

    def forward(self, x):
        x = self.quanv(x) 
        x = torch.relu(self.conv(x))
        x = x.view(-1, config.FC1_INPUT) # Flatten the tensor for the fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x 