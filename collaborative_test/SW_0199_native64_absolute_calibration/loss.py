"""Absolute affinity calibration with equal class/pair and positive/negative mass."""
import torch
from collaborative_test.SW_0179_qce_seed_replication.train import _vectorized_class_loss


def absolute_calibration(q, labels):
    if q.ndim!=2 or q.shape[0]!=q.shape[1] or labels.shape!=(q.shape[0],):
        raise ValueError('square Q and one label per patch required')
    if labels.dtype!=torch.int64 or labels.device!=q.device or not q.is_floating_point():
        raise ValueError('floating Q and int64 labels on same device required')
    if not torch.isfinite(q).all() or (q<0).any() or (q>1).any() or (labels<0).any():
        raise ValueError('finite probability Q and nonnegative labels required')
    classes,inverse,counts=torch.unique(labels,sorted=True,return_inverse=True,return_counts=True)
    if len(classes)<2:return q.sum()*0
    membership=torch.nn.functional.one_hot(inverse,len(classes)).to(q.dtype)
    counts=counts.to(q.dtype);safe=q.clamp(1e-6,1-1e-6)
    log_pos=safe.log();log_neg=torch.log1p(-safe)
    block_pos=membership.T@log_pos@membership
    block_neg=membership.T@log_neg@membership
    # Remove every diagonal self-pair from the same-class numerator.
    diag=torch.zeros_like(counts).scatter_add(0,inverse,log_pos.diagonal())
    valid=counts>1
    positive=-((block_pos.diagonal()[valid]-diag[valid])/(counts[valid]*(counts[valid]-1))).mean() if valid.any() else q.sum()*0
    cross=~torch.eye(len(classes),dtype=torch.bool,device=q.device)
    negative=-(block_neg/(counts[:,None]*counts[None,:]))[cross].mean()
    # Singleton-only scenes contain no positive pairs; retain full negative mass.
    return .5*(positive+negative) if valid.any() else negative


def calibrated_objective(q,labels,*,calibrated):
    loss=_vectorized_class_loss(q,labels,temperature=.1)
    return loss+absolute_calibration(q,labels) if calibrated else loss
