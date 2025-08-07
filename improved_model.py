"""
Improved quantum-classical hybrid model with better information preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from improved_quantum_circuit import ImprovedQuanvLayer

class AttentionGate(nn.Module):
    """
    Attention mechanism to preserve important features before quantum processing.
    """
    def __init__(self, in_channels):
        super(AttentionGate, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


class ResidualQuanvBlock(nn.Module):
    """
    Quantum layer with residual connections to preserve information flow.
    """
    def __init__(self, in_channels, n_qubits=4, n_layers=2):
        super(ResidualQuanvBlock, self).__init__()
        self.quanv = ImprovedQuanvLayer(n_qubits, n_layers, 'hardware_efficient')
        self.out_channels = in_channels * n_qubits
        
        # 1x1 conv for channel matching in residual connection
        self.residual_conv = nn.Conv2d(in_channels, self.out_channels, 
                                       kernel_size=1, stride=2)
        self.norm = nn.GroupNorm(min(8, self.out_channels), self.out_channels)
        
    def forward(self, x):
        # Quantum processing path
        quantum_out = self.quanv(x)
        
        # Residual path (downsample to match quantum output size)
        residual = self.residual_conv(x)
        
        # Combine with residual connection
        out = quantum_out + residual
        return self.norm(F.relu(out))


class ImprovedQuanvNet(nn.Module):
    """
    Enhanced hybrid model with better gradient flow and information preservation.
    """
    def __init__(self, n_qubits=4, num_classes=44, dropout_rate=0.3):
        super(ImprovedQuanvNet, self).__init__()
        
        # Stage 1: Initial feature extraction with attention
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 32x32
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        
        # Attention gate before quantum processing
        self.attention = AttentionGate(8)
        
        # Stage 2: Quantum processing with residual connection
        self.quantum_block = ResidualQuanvBlock(8, n_qubits)
        
        # Stage 3: Classical feature refinement
        quantum_out_channels = 8 * n_qubits
        self.stage3 = nn.Sequential(
            nn.Conv2d(quantum_out_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2)  # 2x2
        )
        
        # Classification head with intermediate supervision
        self.aux_classifier = nn.Linear(quantum_out_channels, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_aux=False):
        # Stage 1: Initial features
        x = self.stage1(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Stage 2: Quantum processing
        quantum_features = self.quantum_block(x)
        
        # Auxiliary output for intermediate supervision
        if return_aux:
            aux_pool = F.adaptive_avg_pool2d(quantum_features, 1)
            aux_out = self.aux_classifier(aux_pool.flatten(1))
        
        # Stage 3: Classical refinement
        x = self.stage3(quantum_features)
        
        # Classification
        x = x.flatten(1)
        out = self.classifier(x)
        
        if return_aux:
            return out, aux_out
        return out


class MultiScaleQuanvNet(nn.Module):
    """
    Multi-scale quantum processing to capture features at different resolutions.
    """
    def __init__(self, n_qubits=4, num_classes=44):
        super(MultiScaleQuanvNet, self).__init__()
        
        # Multi-scale feature extractors
        self.scale1_pre = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True)
        )
        
        self.scale2_pre = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=4, padding=2),  # 8x8
            nn.ReLU(inplace=True)
        )
        
        # Quantum layers for each scale
        self.quanv1 = ImprovedQuanvLayer(n_qubits, n_layers=2)
        self.quanv2 = ImprovedQuanvLayer(n_qubits, n_layers=1)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(4 * n_qubits * 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Process at multiple scales
        scale1 = self.scale1_pre(x)
        scale2 = self.scale2_pre(x)
        
        # Quantum processing
        q1 = self.quanv1(scale1)  # 8x8 output
        q2 = self.quanv2(scale2)  # 4x4 output
        
        # Upsample q2 to match q1 size
        q2_up = F.interpolate(q2, size=q1.shape[-2:], mode='bilinear', align_corners=False)
        
        # Concatenate multi-scale features
        features = torch.cat([q1, q2_up], dim=1)
        
        # Fusion and classification
        features = self.fusion(features)
        features = self.global_pool(features).flatten(1)
        return self.classifier(features)