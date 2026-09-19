"""Exact classical counterpart of the fixed RX/Rot/CNOT-chain/Z map only.

Not an equivalent product formula for V7's interleaved re-uploading circuit.
"""
import numpy as np
import torch


def fixed_bank(seed, filters=4, dtype=torch.float64):
    rng=np.random.RandomState(seed)
    return torch.tensor(np.stack([rng.uniform(-np.pi,np.pi,(4,3)) for _ in range(filters)]),dtype=dtype)


def fixed_expectations(inputs, weights):
    phi,theta=weights[...,0],weights[...,1]
    z=torch.cos(theta)*torch.cos(inputs)-torch.sin(theta)*torch.sin(phi)*torch.sin(inputs)
    return torch.cumprod(z,dim=-1)


def image_patches(images):
    if images.ndim!=4 or images.shape[1:]!=(1,32,32):
        raise ValueError('Expected N×1×32×32 input')
    # Matches historical nested row/column loops and row-major patch flattening.
    return images[:,0].reshape(-1,16,2,16,2).permute(0,1,3,2,4).reshape(len(images),256,4)


def fixed_features(images, weights, output_dtype=torch.float32):
    patches=image_patches(images).to(dtype=weights.dtype,device=weights.device)
    outputs=[fixed_expectations(patches,w).reshape(-1,16,16,4).permute(0,3,1,2) for w in weights]
    return torch.cat(outputs,dim=1).to(output_dtype)
