"""
Ablation Models for Paper — Classical baselines using V7 architecture.

Four variants:
  A) ClassicalBaselineNet        — Conv2d(4,16, k=2, s=2) replaces quantum layer
  D) ParamMatchedLinearNet       — 25-param linear layer replaces quantum layer
  B) NonTrainableQuantumClassicalNet — Henderson-style: trains on pre-computed quantum features

All share identical post-processing with EnhancedQuanvNet (V7).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------
# Shared building blocks (identical to V7)
# -----------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(min(4, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(4, out_channels), out_channels)

    def forward(self, x):
        residual = x
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.gelu(out + residual)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
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


def _make_preprocessing():
    """V7 identical preprocessing: 32x32x1 -> 8x8x4"""
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # 32->16
        nn.GroupNorm(4, 8),
        nn.GELU(),
        ResidualBlock(8, 8),
        nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),   # 16->8
        nn.GroupNorm(2, 4),
        nn.GELU()
    )


def _make_postprocessing(in_channels=16):
    """V7 identical postprocessing: 4x4xC -> 44 logits"""
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
        nn.GroupNorm(8, 32),
        nn.GELU(),
        ResidualBlock(32, 32),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.GroupNorm(8, 64),
        nn.GELU(),
        nn.AdaptiveAvgPool2d(2)
    )


def _make_classifier(num_classes=44):
    """V7 identical classifier head"""
    return nn.Sequential(
        nn.Linear(64 * 4, 128),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(128, 64),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(64, num_classes)
    )


# -----------------
# Experiment A: Classical Conv2d Baseline
# -----------------

class ClassicalConvLayer(nn.Module):
    """Drop-in replacement for TrainableQuanvLayer using Conv2d.
    Input: (B, 4, 8, 8) -> Output: (B, 16, 4, 4)
    Conv2d(4, 16, kernel_size=2, stride=2) = 128 params + 16 bias = 144 params
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 16, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)


class ClassicalBaselineNet(nn.Module):
    """V7 architecture with Conv2d replacing quantum layer."""
    def __init__(self, num_classes=44):
        super().__init__()
        self.pre = _make_preprocessing()
        self.conv_layer = ClassicalConvLayer()
        self.attention = ChannelAttention(16)
        self.skip_weight = nn.Parameter(torch.tensor(0.1))
        self.skip_adapter = nn.Conv2d(4, 16, kernel_size=1, bias=False)
        self.post = _make_postprocessing(16)
        self.classifier = _make_classifier(num_classes)

    def forward(self, x):
        pre_features = self.pre(x)
        features = self.conv_layer(pre_features)
        features = self.attention(features)
        pre_down = F.adaptive_avg_pool2d(pre_features, features.shape[2:])
        features = features + self.skip_weight * self.skip_adapter(pre_down)
        features = self.post(features)
        return self.classifier(features.reshape(features.size(0), -1))


# -----------------
# Experiment D: Parameter-Matched Linear (25 params)
# -----------------

class ParamMatchedLinearLayer(nn.Module):
    """Drop-in replacement using a linear layer with ~25 trainable parameters.
    Input: (B, 4, 8, 8) -> Output: (B, 16, 4, 4)

    Strategy: Apply a small linear map per 2x2 patch (same as quantum).
    Each patch: 4 values -> 4 output values via Linear(4, 4) = 16 weights + 4 bias = 20 params
    Plus 1 gradient_scale + 4 extra = 25 total.
    """
    def __init__(self):
        super().__init__()
        # 4 input features per patch -> 4 output per patch (like quantum circuit)
        self.linear = nn.Linear(4, 4, bias=True)  # 20 params
        self.gradient_scale = nn.Parameter(torch.ones(1) * 0.1)  # 1 param
        # Extra: 4 learnable bias per output channel
        self.channel_bias = nn.Parameter(torch.zeros(4))  # 4 params
        # Total: 20 + 1 + 4 = 25 params (exactly matches quantum)

    def forward(self, x):
        # Extract 2x2 patches like quantum layer does
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (B, 4, 4, 4, 2, 2)
        batch_size, channels, out_h, out_w, _, _ = patches.size()

        # Reshape: (B*patches, 4)
        patches = patches.reshape(batch_size, channels, out_h * out_w, -1)
        patches = patches.permute(0, 2, 1, 3).reshape(-1, 4)

        # Linear processing with gradient scaling
        out = self.linear(patches) * self.gradient_scale + self.channel_bias

        # Reshape back: (B, 16, out_h, out_w)
        out = out.view(batch_size, out_h * out_w, channels, 4)
        out = out.permute(0, 2, 3, 1)
        out = out.reshape(batch_size, channels * 4, out_h, out_w)
        return out


class ParamMatchedLinearNet(nn.Module):
    """V7 architecture with 25-param linear layer replacing quantum layer."""
    def __init__(self, num_classes=44):
        super().__init__()
        self.pre = _make_preprocessing()
        self.linear_layer = ParamMatchedLinearLayer()
        self.attention = ChannelAttention(16)
        self.skip_weight = nn.Parameter(torch.tensor(0.1))
        self.skip_adapter = nn.Conv2d(4, 16, kernel_size=1, bias=False)
        self.post = _make_postprocessing(16)
        self.classifier = _make_classifier(num_classes)

    def forward(self, x):
        pre_features = self.pre(x)
        features = self.linear_layer(pre_features)
        features = self.attention(features)
        pre_down = F.adaptive_avg_pool2d(pre_features, features.shape[2:])
        features = features + self.skip_weight * self.skip_adapter(pre_down)
        features = self.post(features)
        return self.classifier(features.reshape(features.size(0), -1))


# -----------------
# Experiment B: Henderson-Style Non-Trainable Quantum
# Classical network trained on pre-computed quantum features
# -----------------

class NonTrainableQuantumClassicalNet(nn.Module):
    """Classical network trained on Henderson-style pre-computed quantum features.

    Henderson et al. (2020) approach: quantum circuit applied ONCE to all images
    as preprocessing, results cached to disk, then classical network trains on
    cached quantum features. No quantum computation during training.

    Input: pre-computed quantum features (B, n_filters*4, 16, 16)
    where n_filters random quantum circuits each produce 4 channels from 2x2 patches.
    """
    def __init__(self, in_channels=16, num_classes=44):
        super().__init__()
        # Spatial reduction: 16x16 -> 4x4 (matching V7 quantum output spatial dims)
        # Lightweight: in_channels -> 8 -> 16 to keep param count near V7's ~87,800
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1),  # 16->8
            nn.GroupNorm(4, 8),
            nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 8->4
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        # Same post-processing as V7 from (16, 4, 4) onward
        self.attention = ChannelAttention(16)
        self.post = _make_postprocessing(16)
        self.classifier = _make_classifier(num_classes)

    def forward(self, x):
        x = self.reduce(x)
        x = self.attention(x)
        x = self.post(x)
        return self.classifier(x.reshape(x.size(0), -1))
