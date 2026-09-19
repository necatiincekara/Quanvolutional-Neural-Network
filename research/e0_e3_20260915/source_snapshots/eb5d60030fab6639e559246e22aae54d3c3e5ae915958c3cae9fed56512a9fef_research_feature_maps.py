"""Small E2 common raw-patch pipeline; no V8+ architecture is implemented."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.classical_quantum_controls import fixed_bank,fixed_features,image_patches

FAMILIES=('analytic_fixed','random_conv','orthogonal','rff','random_mlp','polynomial','learned_conv')


class PatchMap(nn.Module):
    def __init__(self,family,seed):
        super().__init__();self.family=family
        if family not in FAMILIES:raise ValueError(family)
        g=torch.Generator().manual_seed(seed)
        if family=='analytic_fixed':
            self.register_buffer('bank',fixed_bank(seed));return
        w=torch.randn(4,16,generator=g)/2
        if family=='orthogonal':
            w=torch.cat([torch.linalg.qr(torch.randn(4,4,generator=g))[0] for _ in range(4)],1)
        b=torch.rand(16,generator=g)*2-1
        if family=='polynomial':
            self.register_buffer('rotation',torch.linalg.qr(torch.randn(4,4,generator=g))[0]);return
        if family=='learned_conv':
            self.w=nn.Parameter(w);self.b=nn.Parameter(b)
        else:
            self.register_buffer('w',w);self.register_buffer('b',b)
        if family=='random_mlp':self.register_buffer('w2',torch.randn(16,16,generator=g)/4)

    def forward(self,images):
        if self.family=='analytic_fixed':return fixed_features(images,self.bank)
        x=image_patches(images)-.5
        if self.family=='polynomial':
            v=x@self.rotation
            feats=[torch.ones_like(v[...,0])]
            for mask in range(1,16):
                feats.append(v[...,[(mask>>i)&1==1 for i in range(4)]].prod(-1))
            y=torch.stack(feats,-1)
        elif self.family=='rff':y=math.sqrt(2)*torch.cos(x@self.w+self.b*math.pi)
        elif self.family=='random_mlp':y=torch.tanh(torch.tanh(x@self.w+self.b)@self.w2)
        else:y=torch.tanh(x@self.w+self.b)
        return y.reshape(-1,16,16,16).permute(0,3,1,2).contiguous()


def pooled(features):return F.adaptive_avg_pool2d(features,(4,4)).flatten(1)
