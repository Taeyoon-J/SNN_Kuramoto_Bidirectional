"""Class-balanced spatial constraints on actual production component spikes."""
from __future__ import annotations
import numpy as np
import torch
from collaborative_test.SW_0179_qce_seed_replication.train import _vectorized_class_loss

POSITIVE_Q_MARGIN=.55
NEGATIVE_Q_MARGIN=.45
POSITIVE_RHO_MARGIN=POSITIVE_Q_MARGIN**.25

def spatial_edges(grid,device):
    if type(grid) is not int or grid<2:raise ValueError('grid must be an integer >=2')
    ids=torch.arange(grid*grid,device=device).reshape(grid,grid)
    return (torch.cat((ids[:,:-1].reshape(-1),ids[:-1,:].reshape(-1))),
        torch.cat((ids[:,1:].reshape(-1),ids[1:,:].reshape(-1))))

def connectivity_loss(q,rho,labels,*,grid=32):
    n=grid*grid
    if q.shape!=(n,n) or rho.shape!=(4,n,n) or labels.shape!=(n,):raise ValueError('native geometry mismatch')
    if labels.dtype==torch.bool or labels.dtype.is_floating_point or labels.dtype.is_complex:raise TypeError('integer labels required')
    if q.device!=rho.device or q.device!=labels.device:raise ValueError('device mismatch')
    if not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(rho).all()):raise ValueError('nonfinite affinities')
    if bool((labels<0).any()):raise ValueError('negative modal ID')
    i,j=spatial_edges(grid,q.device);a,b=labels[i],labels[j]
    positive=a==b;negative=~positive
    terms=[]
    if bool(positive.any()):
        penalties=(POSITIVE_RHO_MARGIN-rho[:,i[positive],j[positive]]).clamp_min(0).square().mean(0)
        groups=a[positive]
        terms.append(torch.stack([penalties[groups==cls].mean() for cls in torch.unique(groups)]).mean())
    if bool(negative.any()):
        penalties=(q[i[negative],j[negative]]-NEGATIVE_Q_MARGIN).clamp_min(0).square()
        groups=torch.stack((torch.minimum(a[negative],b[negative]),torch.maximum(a[negative],b[negative])),1)
        _,inverse=torch.unique(groups,dim=0,return_inverse=True)
        terms.append(torch.stack([penalties[inverse==cls].mean() for cls in torch.unique(inverse)]).mean())
    return torch.stack(terms).mean() if terms else q.sum()*0+rho.sum()*0

def loss_for_image(q,rho,labels):
    return _vectorized_class_loss(q,labels,temperature=.1)+connectivity_loss(q,rho,labels)
