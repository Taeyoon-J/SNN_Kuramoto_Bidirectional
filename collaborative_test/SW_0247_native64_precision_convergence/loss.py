"""Increase only class-balanced within-object variance; keep teacher center/anchor terms."""
import numpy as np
import torch
from collaborative_test.SW_0222_native64_frequency_teacher_denoising.teacher_loss import teacher_objective


def groups_from_labels(labels):
    y=np.asarray(labels)
    if y.ndim!=3 or not y.size or y.dtype.kind not in 'iu' or np.any(y<0):
        raise ValueError('nonempty integer TRAIN masks required')
    return tuple(tuple(torch.from_numpy(np.flatnonzero(row.reshape(-1)==k).astype(np.int64))
        for k in np.unique(row)) for row in y)


def groups_to_device(groups,device):
    return tuple(tuple(index.to(device) for index in image) for image in groups)


def within_variance(frequency,groups):
    if frequency.ndim!=3 or frequency.shape[-1]!=4 or len(groups)!=len(frequency):
        raise ValueError('aligned actual four-component frequency groups required')
    images=[]
    for values,classes in zip(frequency,groups):
        if not classes:raise ValueError('each TRAIN image must have a class')
        terms=[]
        for index in classes:
            if index.ndim!=1 or not index.numel():raise ValueError('nonempty class indices required')
            points=values.index_select(0,index.to(frequency.device))
            terms.append((points-points.mean(0,keepdim=True)).square().mean())
        images.append(torch.stack(terms).mean())
    return torch.stack(images).mean()


def objective(gamma,frequency,targets,groups,within_coefficient):
    if within_coefficient not in (1.,16.):raise ValueError('registered matched-control or within16 arm required')
    base,parts=teacher_objective(gamma,frequency,targets)
    within=within_variance(frequency,groups)
    # Control is the exact original computation, not a reconstructed decomposition.
    loss=base if within_coefficient==1. else base+15.*within
    return loss,dict(parts,within_variance=within)
