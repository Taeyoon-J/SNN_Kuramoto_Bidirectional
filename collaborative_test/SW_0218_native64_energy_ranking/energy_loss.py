"""Rank actual production partition energy against a feasible GT reference.

The before-step partition, support, reference and energy scale are frozen for
the entire optimizer proposal. GT is TRAIN-only. This is a ranking surrogate,
not a differentiable ARI or a bound on the discrete optimum.
"""
import hashlib
import numpy as np
import torch
from collaborative_test.SW_0162_connected_signed_partition import connected_signed as native
from collaborative_test.SW_0217_native64_energy_refinement.refine import connected


def pair_energy(q,adjacency,gain):
    if q.ndim!=2 or q.shape[0]!=q.shape[1] or adjacency.shape!=q.shape:
        raise ValueError('matching square native Q and adjacency required')
    if q.device!=adjacency.device or gain.numel()!=1 or gain.device!=q.device:
        raise ValueError('scalar native gain and one device required')
    if not all(bool(torch.isfinite(v).all()) for v in (q,adjacency,gain)) or float(gain.detach())<=0:
        raise ValueError('finite evidence and positive gain required')
    if q.numel() and (float(q.detach().min()) < -1e-7 or float(q.detach().max()) > 1+1e-7):
        raise ValueError('Q outside probability range')
    q64=q.double();a64=adjacency.double();g=gain.double()
    sym=.5*(q64+q64.T);sym=sym.clamp(native.Q_EPS,1-native.Q_EPS)
    energy=(1+native.NATIVE_NODES*(a64+a64.T)/(2*g))*torch.log(sym/(1-sym))
    energy=.5*(energy+energy.T)
    return energy-torch.diag_embed(torch.diagonal(energy))


def make_context(predicted,truth,q,adjacency,gain):
    pred=np.asarray(predicted);gt=np.asarray(truth)
    if pred.ndim!=2 or gt.shape!=pred.shape or pred.dtype.kind not in 'iu' or gt.dtype.kind not in 'iu' or (pred<0).any() or (gt<0).any():
        raise ValueError('matching nonnegative integer TRAIN grids required')
    if q.shape!=(pred.size,pred.size) or adjacency.shape!=q.shape:
        raise ValueError('full native evidence must match TRAIN grid')
    flat=pred.reshape(-1);target=gt.reshape(-1);support=np.flatnonzero(flat>0)
    reference=np.zeros(pred.size,np.int64);label=0
    # GT0 inside predicted foreground remains a positive reference group. Never
    # drop a false-positive patch or supply a GT count to the production readout.
    for group in np.unique(target[support]):
        for piece in connected(support[target[support]==group],pred.shape):
            label+=1;reference[piece]=label
    a=flat[support];b=reference[support]
    difference=((a[:,None]==a[None,:]).astype(np.int8)
                -(b[:,None]==b[None,:]).astype(np.int8))
    i=torch.as_tensor(support,device=q.device)
    with torch.no_grad():
        energy=pair_energy(q[i[:,None],i],adjacency[i[:,None],i],gain)
        changed=torch.as_tensor(np.abs(difference),device=q.device)
        scale=float((energy.abs()*changed).sum().cpu())
    pairs=int(np.abs(difference).sum())//2
    digest=hashlib.sha256()
    for value in (pred,reference,support,difference):digest.update(np.ascontiguousarray(value).tobytes())
    # Machine floor is only for zero-energy disagreements, not a fitted margin.
    denominator=max(scale,2*pairs*native.Q_EPS)
    return dict(support=support,difference=difference,scale=denominator,
                predicted_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),
                reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                context_sha256=digest.hexdigest(),disagreement_pairs=pairs,
                reference_groups=label,source_foreground_patches=len(support))


def ranking_loss(q,adjacency,gain,context):
    support=context['support'];difference=context['difference']
    if difference.shape!=(len(support),len(support)) or context['scale']<0:
        raise ValueError('invalid frozen ranking context')
    if not context['disagreement_pairs']:
        return (q.sum()+adjacency.sum()+gain.sum())*0
    if not context['scale']>0:raise ValueError('nonzero disagreement requires positive frozen scale')
    i=torch.as_tensor(support,device=q.device)
    energy=pair_energy(q[i[:,None],i],adjacency[i[:,None],i],gain)
    d=torch.as_tensor(difference,device=q.device,dtype=energy.dtype)
    # Numerator and denominator both count unordered pairs twice. At the
    # initial point the energy term is in [-1,1]; unit margin is fixed.
    return torch.relu(1+(energy*d).sum()/context['scale'])


def context_record(context):
    return {key:context[key] for key in ('predicted_sha256','reference_sha256',
        'context_sha256','disagreement_pairs','reference_groups','source_foreground_patches','scale')}
