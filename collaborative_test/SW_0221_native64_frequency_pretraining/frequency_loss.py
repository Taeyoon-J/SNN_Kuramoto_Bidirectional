"""Class-balanced compactness of the actual sensory effective frequencies."""
import math
import numpy as np
import torch


def make_targets(labels):
    labels=np.asarray(labels)
    if labels.ndim!=3 or min(labels.shape)<1 or labels.dtype.kind not in 'iu' or (labels<0).any():
        raise ValueError('nonnegative integer [B,H,W] TRAIN labels required')
    classes=[np.unique(row) for row in labels];width=max(map(len,classes));batch,height,cols=labels.shape
    membership=np.zeros((batch,width,height*cols),bool);adjacent=np.zeros((batch,width,width),bool)
    for index,(grid,ids) in enumerate(zip(labels,classes)):
        flat=grid.reshape(-1);membership[index,:len(ids)]=ids[:,None]==flat[None,:]
        codes=np.searchsorted(ids,grid)
        for left,right in ((codes[:,:-1],codes[:,1:]),(codes[:-1,:],codes[1:,:])):
            different=left!=right;a=np.minimum(left[different],right[different]);b=np.maximum(left[different],right[different])
            adjacent[index,a,b]=True
    return dict(membership=torch.from_numpy(membership),adjacent=torch.from_numpy(adjacent))


def effective_frequency(core,gamma):
    gain=core.kuramoto.freq_gain
    if gain is None or gain.numel()!=1 or not bool(torch.isfinite(gain).all()) or float(gain.detach().abs())<=1e-6:
        raise ValueError('actual nonzero finite sensory frequency gain required')
    drive=core.gamma_to_drive(gamma,core.gamma_channel_proj,core.gamma_phase_gain)
    omega=core.kuramoto.omega
    if drive.ndim!=3 or omega.shape!=drive.shape[1:]:raise ValueError('actual per-node/component drive and omega required')
    return (omega.unsqueeze(0)+gain*drive)/(gain.abs()*math.pi)


def frequency_objective(frequency,targets):
    if frequency.ndim!=3 or min(frequency.shape)<1 or not bool(torch.isfinite(frequency).all()):
        raise ValueError('finite actual normalized frequency [B,N,D] required')
    membership=targets['membership'].to(device=frequency.device,dtype=frequency.dtype)
    adjacent=targets['adjacent'].to(device=frequency.device,dtype=frequency.dtype)
    if membership.ndim!=3 or membership.shape[0]!=frequency.shape[0] or membership.shape[2]!=frequency.shape[1] or adjacent.shape!=(*membership.shape[:2],membership.shape[1]):
        raise ValueError('membership/adjacency must match every phase-driving patch')
    if not bool((membership.sum(1)==1).all()):raise ValueError('every patch must belong to exactly one TRAIN class')
    counts=membership.sum(2);active=(counts>0).to(frequency.dtype)
    mean=torch.bmm(membership,frequency)/counts.clamp_min(1).unsqueeze(-1)
    second=torch.bmm(membership,frequency.square())/counts.clamp_min(1).unsqueeze(-1)
    variance=(second-mean.square()).clamp_min(0).mean(2)
    within=(variance*active).sum(1)/active.sum(1).clamp_min(1)
    distance=(mean.unsqueeze(2)-mean.unsqueeze(1)).square().mean(3)
    between=(torch.relu(1-distance).square()*adjacent).sum((1,2))/adjacent.sum((1,2)).clamp_min(1)
    return (within+between).mean(),dict(within=within.mean(),adjacent_separation=between.mean())
