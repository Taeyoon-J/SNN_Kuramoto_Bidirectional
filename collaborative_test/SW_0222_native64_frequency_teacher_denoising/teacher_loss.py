"""Denoise towards source object means without inventing a frequency margin."""
import numpy as np
import torch


def make_teacher(gamma,frequency,labels):
    gamma=np.asarray(gamma);frequency=np.asarray(frequency);labels=np.asarray(labels)
    if labels.ndim!=3 or labels.dtype.kind not in 'iu' or (labels<0).any():raise ValueError('integer TRAIN patch labels required')
    batch=len(labels);nodes=int(np.prod(labels.shape[1:]))
    if gamma.ndim!=3 or gamma.shape[0]!=batch or gamma.shape[2]!=nodes or frequency.ndim!=3 or frequency.shape[:2]!=(batch,nodes):raise ValueError('aligned source gamma and actual frequency required')
    if gamma.dtype.kind!='f' or frequency.dtype.kind!='f' or not np.isfinite(gamma).all() or not np.isfinite(frequency).all():raise ValueError('finite source floating arrays required')
    centers=np.empty_like(frequency);weights=np.empty((batch,nodes),dtype=frequency.dtype)
    for index,grid in enumerate(labels):
        flat=grid.reshape(-1);classes=np.unique(flat)
        for k in classes:
            take=flat==k;centers[index,take]=frequency[index,take].mean(0)
            weights[index,take]=1./(len(classes)*int(take.sum()))
    return dict(gamma_reference=torch.from_numpy(gamma.copy()),frequency_centers=torch.from_numpy(centers),class_weights=torch.from_numpy(weights))


def teacher_objective(gamma,frequency,targets):
    old=targets['gamma_reference'].to(device=gamma.device,dtype=gamma.dtype)
    centers=targets['frequency_centers'].to(device=frequency.device,dtype=frequency.dtype)
    weights=targets['class_weights'].to(device=frequency.device,dtype=frequency.dtype)
    if gamma.shape!=old.shape or frequency.shape!=centers.shape or weights.shape!=frequency.shape[:2] or gamma.shape[0]!=frequency.shape[0] or gamma.shape[2]!=frequency.shape[1]:raise ValueError('student and frozen teacher geometries differ')
    if old.requires_grad or centers.requires_grad or weights.requires_grad:raise ValueError('teacher must be detached')
    if not bool(torch.isfinite(gamma).all() and torch.isfinite(frequency).all()):raise ValueError('finite student tensors required')
    if not bool((weights>0).all()) or not torch.allclose(weights.sum(1),torch.ones(len(weights),device=weights.device,dtype=weights.dtype)):raise ValueError('class-balanced image weights required')
    denoising=((frequency-centers).square().mean(2)*weights).sum(1).mean()
    anchor=(gamma-old).square().mean()
    return denoising+anchor,dict(frequency_denoising=denoising,gamma_anchor=anchor)
