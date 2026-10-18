"""Same real-valued soft geodesic; bounded float64 exponential/GEMM products."""
import math
import types
import numpy as np
import torch
from torch.nn import functional as F


def geodesic_matmul(z, euclidean, contrast, *, steps=3, radius=1.5,
                    temperature=.5, cap=16., log4=True):
    if z.ndim != 3 or z.dtype not in (torch.float32, torch.float64):
        raise ValueError('float32/64 z[B,N,D] required')
    batch, n, _ = z.shape
    if (type(steps) is not int or not 0 <= steps <= 3 or not 1 <= n <= 4096
            or not all(math.isfinite(float(v)) and v > 0 for v in (radius, temperature, cap))
            or cap/temperature > 256):
        raise ValueError('outside proved finite GEMM domain')
    if euclidean.ndim == 2: euclidean = euclidean.unsqueeze(0)
    if (euclidean.ndim != 3 or euclidean.shape[0] not in (1, batch)
            or tuple(euclidean.shape[1:]) != (n,n)):
        raise ValueError('Euclidean geometry mismatch')
    euclidean = euclidean.to(device=z.device, dtype=z.dtype)
    contrast = torch.as_tensor(contrast, device=z.device, dtype=z.dtype)
    if (contrast.numel() != 1 or not torch.isfinite(contrast).all()
            or not torch.isfinite(z).all() or not torch.isfinite(euclidean).all()
            or (euclidean < 0).any()):
        raise ValueError('finite inputs and nonnegative distances required')
    similarity = torch.bmm(z, z.transpose(1,2)).clamp(-1.,1.)
    step = euclidean * (1. + F.softplus(contrast) * (1.-similarity).square())
    distance = torch.where(euclidean <= radius, step, torch.full_like(step, cap))
    # Source step can exceed cap INSIDE radius. Preserve it exactly: source
    # applies cap only at the end. Use double products rather than clipping
    # source edges or relying on float32 subnormals for exp(-49.5/.5).
    if not torch.isfinite(distance).all() or (distance/temperature > 256).any():
        raise ValueError('initial distance outside proved finite exponential domain')
    if not torch.equal(torch.diagonal(euclidean,dim1=-2,dim2=-1),
                       torch.zeros_like(torch.diagonal(euclidean,dim1=-2,dim2=-1))):
        raise ValueError('zero diagonal required by production distance contract')
    correction = temperature*torch.log(torch.tensor(4.,device=z.device,dtype=z.dtype)) if log4 else 0.
    for _ in range(steps):
        kernel = torch.exp(-distance.to(torch.float64)/temperature)
        product = torch.bmm(kernel,kernel)
        if (product <= 0).any() or not torch.isfinite(product).all():
            raise FloatingPointError('GEMM underflow/overflow; do not substitute an approximation')
        relaxed = (-temperature*product.log()+correction).to(distance.dtype)
        distance = torch.minimum(distance,relaxed)
    return distance.clamp(max=cap)


def install(graph):
    from collaborative_test.SW_0143_native32_selective_geodesic.geodesic_operator import inverse_softplus
    def distance(self,z,euclidean):
        fixed=torch.as_tensor(inverse_softplus(8.),device=z.device,dtype=z.dtype)
        return geodesic_matmul(z,euclidean,fixed,steps=self.geodesic_steps,
            radius=self.geodesic_radius,temperature=self.geodesic_temperature,cap=self.geodesic_cap,log4=True)
    graph._geodesic_distance=types.MethodType(distance,graph)
    return graph
