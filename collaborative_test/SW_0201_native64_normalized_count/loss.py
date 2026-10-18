"""Supervised class-count regularization of native spike-Q, with unchanged QCE."""
import numpy as np  # Initialize NumPy before Torch on Windows.
import torch
from collaborative_test.SW_0179_qce_seed_replication.train import _vectorized_class_loss
from collaborative_test.SW_0197_native64_baseline_replication.count_proxy_review import normalized_proxy


def count_penalty(q,labels):
    if q.ndim!=2 or q.shape[0]!=q.shape[1] or q.shape[0]==0:raise ValueError('nonempty square Q required')
    if labels.dtype!=torch.int64 or labels.ndim!=1 or labels.shape[0]!=q.shape[0] or labels.device!=q.device:
        raise ValueError('one int64 class per patch on the same device required')
    if (labels<0).any():raise ValueError('negative class label')
    effective=normalized_proxy(q)
    classes=torch.unique(labels).numel()
    if classes<1:raise ValueError('empty class target')
    target=q.new_tensor(float(classes))
    return (effective.clamp_min(1e-6).log()-target.log()).square()


def count_calibrated_objective(q,labels):
    # Unit weight fixed before GPU results; no hard count imposed at inference.
    penalty=count_penalty(q,labels)
    return _vectorized_class_loss(q,labels,temperature=.1)+penalty
