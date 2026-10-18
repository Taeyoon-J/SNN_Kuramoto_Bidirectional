"""Supervise signed component affinities before the hard positive product."""
import math
import numpy as np
import torch
from collaborative_test.SW_0179_qce_seed_replication.train import _vectorized_class_loss

def component_objective(q,rho,labels,*,temperature=.1):
    if q.ndim!=2 or q.shape[0]!=q.shape[1] or rho.shape!=(4,*q.shape):raise ValueError('native four-component square affinity required')
    if q.device!=rho.device or q.dtype!=rho.dtype:raise ValueError('component and product affinity device/dtype differ')
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('finite positive temperature required')
    if not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(rho).all()):raise ValueError('finite affinities required')
    return _vectorized_class_loss(q,labels,temperature=temperature)+torch.stack([_vectorized_class_loss(row,labels,temperature=temperature) for row in rho]).mean()
