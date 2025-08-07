"""
Improved quantum circuit designs with enhanced expressivity and gradient flow.
"""

import pennylane as qml
import torch
import torch.nn as nn

def create_improved_quantum_circuit(n_qubits=4, n_layers=3, device='lightning.gpu'):
    """
    Creates a more expressive quantum circuit with multiple layers and better entanglement.
    
    Key improvements:
    1. Multiple layers for increased expressivity
    2. Data re-uploading between layers
    3. All-to-all entanglement for better feature correlation
    4. Trainable input scaling for amplitude control
    """
    dev = qml.device(device, wires=n_qubits)
    
    @qml.qnode(dev, interface='torch', diff_method='adjoint')
    def circuit(inputs, weights, input_scaling):
        """
        Enhanced quantum circuit with data re-uploading and deeper architecture.
        
        Args:
            inputs: Input features (n_qubits,)
            weights: Trainable parameters (n_layers, n_qubits, 3)
            input_scaling: Trainable scaling factors for inputs (n_qubits,)
        """
        # Scale inputs to control amplitude
        scaled_inputs = inputs * torch.sigmoid(input_scaling)
        
        for layer in range(n_layers):
            # Data re-uploading: encode inputs at each layer
            qml.AngleEmbedding(scaled_inputs, wires=range(n_qubits))
            
            # Parameterized rotation layer
            for i in range(n_qubits):
                qml.Rot(weights[layer, i, 0], 
                       weights[layer, i, 1], 
                       weights[layer, i, 2], wires=i)
            
            # Enhanced entanglement pattern
            if layer < n_layers - 1:  # Skip entanglement on last layer
                # All-to-all connectivity for better expressivity
                for i in range(n_qubits):
                    for j in range(i+1, n_qubits):
                        qml.CRZ(weights[layer, i, 0] * 0.1, wires=[i, j])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return circuit


def create_hardware_efficient_circuit(n_qubits=4, n_layers=2, device='lightning.gpu'):
    """
    Hardware-efficient ansatz with better gradient properties.
    
    This circuit uses:
    1. RY-RZ rotations (avoids barren plateaus better than full Rot gates)
    2. Ring connectivity for entanglement
    3. Parameterized entangling gates
    """
    dev = qml.device(device, wires=n_qubits)
    
    @qml.qnode(dev, interface='torch', diff_method='adjoint')
    def circuit(inputs, weights):
        """
        Hardware-efficient circuit with controlled depth.
        
        Args:
            inputs: Input features (n_qubits,)
            weights: Trainable parameters (n_layers * 2 + 1, n_qubits, 2)
        """
        # Initial encoding with RY rotations (better for gradients)
        for i in range(n_qubits):
            qml.RY(inputs[i] * torch.pi, wires=i)
        
        weight_idx = 0
        for layer in range(n_layers):
            # Single-qubit rotation layer
            for i in range(n_qubits):
                qml.RY(weights[weight_idx, i, 0], wires=i)
                qml.RZ(weights[weight_idx, i, 1], wires=i)
            weight_idx += 1
            
            # Entangling layer with parameterized CRZ gates
            for i in range(n_qubits):
                next_qubit = (i + 1) % n_qubits
                qml.CRZ(weights[weight_idx, i, 0], wires=[i, next_qubit])
            weight_idx += 1
        
        # Final rotation layer
        for i in range(n_qubits):
            qml.RY(weights[weight_idx, i, 0], wires=i)
            qml.RZ(weights[weight_idx, i, 1], wires=i)
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return circuit


class ImprovedQuanvLayer(nn.Module):
    """
    Improved quantum convolutional layer with multiple enhancements.
    """
    def __init__(self, n_qubits=4, n_layers=2, circuit_type='hardware_efficient'):
        super(ImprovedQuanvLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        if circuit_type == 'expressive':
            circuit = create_improved_quantum_circuit(n_qubits, n_layers)
            weight_shapes = {
                "weights": (n_layers, n_qubits, 3),
                "input_scaling": (n_qubits,)
            }
        else:  # hardware_efficient
            circuit = create_hardware_efficient_circuit(n_qubits, n_layers)
            weight_shapes = {
                "weights": (n_layers * 2 + 1, n_qubits, 2)
            }
        
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        
        # Add learnable output scaling for better gradient flow
        self.output_scale = nn.Parameter(torch.ones(n_qubits))
        self.output_bias = nn.Parameter(torch.zeros(n_qubits))
    
    def forward(self, x):
        # Same patch extraction as original
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
        batch_size, channels, out_h, out_w, _, _ = patches.size()
        
        patches = patches.reshape(batch_size, channels, out_h * out_w, -1)
        patches = patches.permute(0, 2, 1, 3).reshape(-1, self.n_qubits)
        
        # Process through quantum circuit
        processed_patches = self.qlayer(patches)
        
        # Apply learned scaling and bias for better gradient flow
        processed_patches = processed_patches * self.output_scale + self.output_bias
        
        # Reshape back
        final_shape = (batch_size, out_h * out_w, channels, self.n_qubits)
        processed_patches = processed_patches.view(final_shape)
        processed_patches = processed_patches.permute(0, 2, 3, 1)
        processed_patches = processed_patches.reshape(
            batch_size, channels * self.n_qubits, out_h, out_w
        )
        
        return processed_patches.to(x.device)


# Gradient flow monitoring utilities
def analyze_circuit_gradients(circuit, sample_input, sample_weights):
    """
    Analyzes gradient flow through the quantum circuit.
    """
    import torch.autograd as autograd
    
    # Enable gradient computation
    sample_input.requires_grad_(True)
    sample_weights.requires_grad_(True)
    
    # Forward pass
    output = circuit(sample_input, sample_weights)
    
    # Compute gradients
    grads = []
    for out in output:
        grad = autograd.grad(out, sample_weights, 
                            retain_graph=True, 
                            allow_unused=True)[0]
        if grad is not None:
            grads.append(grad.abs().mean().item())
    
    return {
        'mean_gradient': torch.tensor(grads).mean().item(),
        'std_gradient': torch.tensor(grads).std().item(),
        'min_gradient': torch.tensor(grads).min().item(),
        'max_gradient': torch.tensor(grads).max().item()
    }