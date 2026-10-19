"""Denoise source frequencies while preventing relative class-center contraction."""
import torch
from collaborative_test.SW_0228_native64_center_preservation import teacher_loss as parent

CENTER_WEIGHT=1.
SEPARATION_WEIGHT=1.
BASE_KEYS=parent.BASE_KEYS
make_teacher=parent.make_teacher


def relative_separation(frequency,targets):
    labels=targets['class_labels'].to(device=frequency.device)
    source=targets['frequency_centers'].to(device=frequency.device,dtype=frequency.dtype)
    if labels.dtype!=torch.int64 or labels.shape!=frequency.shape[:2] or bool((labels<0).any()):
        raise ValueError('unaltered aligned TRAIN labels required')
    if source.shape!=frequency.shape or not bool(torch.isfinite(source).all()):
        raise ValueError('finite aligned source frequency teacher required')
    losses=[]
    for b in range(len(labels)):
        current=[];reference=[]
        for label in torch.unique(labels[b]):
            take=labels[b]==label
            current.append(frequency[b,take].mean(0));reference.append(source[b,take][0])
        if len(current)<2:
            losses.append(frequency[b].sum()*0.);continue
        current=torch.stack(current);reference=torch.stack(reference)
        i,j=torch.triu_indices(len(current),len(current),offset=1,device=frequency.device)
        actual=(current[i]-current[j]).square().mean(-1)
        natural=(reference[i]-reference[j]).square().mean(-1)
        # Same units as frequency MSE; no arbitrary absolute distance margin.
        deficit=torch.relu(natural-actual)
        losses.append((deficit.square()/natural.clamp_min(torch.finfo(frequency.dtype).eps)).mean())
    return torch.stack(losses).mean()


def teacher_objective(gamma,frequency,targets):
    original,parts=parent.teacher_objective(gamma,frequency,targets,center_weight=CENTER_WEIGHT)
    separation=relative_separation(frequency,targets)
    return original+SEPARATION_WEIGHT*separation,dict(parts,relative_separation=separation)
