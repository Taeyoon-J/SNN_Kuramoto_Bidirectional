"""Chance-corrected foreground pair agreement surrogate, not hard mask ARI."""
import math
import torch
from collaborative_test.SW_0179_qce_seed_replication.train import _vectorized_class_loss

def foreground_pair_ari_loss(q,labels):
    if q.ndim!=2 or q.shape[0]!=q.shape[1] or not q.is_floating_point():raise ValueError('floating square affinity required')
    if labels.shape!=(q.shape[0],) or labels.dtype not in (torch.int32,torch.int64) or labels.device!=q.device:raise ValueError('matching integer labels required')
    if bool((labels<0).any()) or not bool(torch.isfinite(q).all()):raise ValueError('nonnegative labels and finite affinity required')
    if bool((q< -1e-6).any() | (q>1+1e-6).any()):raise ValueError('probability affinity must be in0..1')
    foreground=torch.nonzero(labels>0,as_tuple=False).flatten();n=foreground.numel()
    if n<2:return q.sum()*0
    values=q.index_select(0,foreground).index_select(1,foreground)
    target=labels.index_select(0,foreground)
    i,j=torch.triu_indices(n,n,offset=1,device=q.device)
    p=(.5*(values[i,j].double()+values[j,i].double()))
    same=(target[i]==target[j]).to(torch.float64)
    pairs=float(p.numel());truth=same.sum();predicted=p.sum();joint=(same*p).sum()
    expected=truth*predicted/pairs
    denominator=.5*(truth+predicted)-expected
    # Both singleton partitions or both one-cluster partitions: exact ARI=1.
    if float(denominator.detach())<=1e-15:return q.sum()*0
    return (1-(joint-expected)/denominator).to(q.dtype)

def foreground_pair_objective(q,labels,*,temperature=.1):
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('finite positive temperature required')
    return _vectorized_class_loss(q,labels,temperature=temperature)+foreground_pair_ari_loss(q,labels)
