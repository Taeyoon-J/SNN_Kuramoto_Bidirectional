"""The original all-intermediate-node soft min-plus operator using stable GEMM."""
import torch
from torch.nn import functional as F


def distance_power2(z, euclidean, raw_contrast, *, steps=3, radius=1.5, temperature=.5, cap=16.):
    if z.ndim!=3 or temperature<=0:raise ValueError('invalid graph geometry')
    if euclidean.ndim==2:euclidean=euclidean.unsqueeze(0)
    similarity=torch.bmm(z,z.transpose(1,2)).clamp(-1.,1.)
    edge=euclidean*(1.+F.softplus(raw_contrast)*(1.-similarity).square())
    distance=torch.where(euclidean<=radius,edge,torch.full_like(edge,cap))
    correction=temperature*torch.log(torch.as_tensor(4.,device=z.device,dtype=z.dtype))
    for _ in range(steps):
        # exp(-(Dik+Dkj)/tau) factorizes exactly into two matrices.
        # Detached shifts cancel algebraically; all D derivatives remain live.
        row=distance.amin(-1,keepdim=True).detach()
        column=distance.amin(-2,keepdim=True).detach()
        left=torch.exp((row-distance)/temperature)
        right=torch.exp((column-distance)/temperature)
        mass=torch.bmm(left,right)
        if not bool((mass>0).all()) or not bool(torch.isfinite(mass).all()):
            raise FloatingPointError('geodesic GEMM exceeded its safe numeric domain')
        relaxed=row+column-temperature*mass.log()+correction
        distance=torch.minimum(distance,relaxed)
    return distance.clamp(max=cap)
