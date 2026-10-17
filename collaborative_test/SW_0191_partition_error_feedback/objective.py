"""Class-balanced contrastive penalties on frozen production-partition errors."""
import torch
from collaborative_test.SW_0179_qce_seed_replication.train import _vectorized_class_loss


def _group_mean(values, groups, count):
    # Large BG error groups require stable reductions for strict loss parity.
    stable=values.to(torch.float64)
    total=stable.new_zeros(count).scatter_add_(0,groups,stable)
    sizes=torch.bincount(groups,minlength=count).to(torch.float64)
    valid=sizes>0
    return (total[valid]/sizes[valid]).mean().to(values.dtype)


def partition_error_loss(q,rho,labels,predicted):
    n=labels.numel()
    if q.shape!=(n,n) or rho.shape!=(4,n,n) or labels.shape!=(n,) or predicted.shape!=(n,):
        raise ValueError('affinity/label geometry mismatch')
    if labels.dtype not in (torch.int32,torch.int64) or predicted.dtype not in (torch.int32,torch.int64):
        raise ValueError('integer modal/predicted labels required')
    if any(x.device!=q.device for x in (rho,labels,predicted)):
        raise ValueError('affinity/label devices differ')
    if bool((labels<0).any()) or bool((predicted<0).any()) or not bool(torch.isfinite(q).all() & torch.isfinite(rho).all()):
        raise ValueError('invalid labels or nonfinite affinity')
    _,inverse=torch.unique(labels,sorted=True,return_inverse=True)
    classes=int(inverse.max().item())+1 if n else 0
    i,j=torch.triu_indices(n,n,offset=1,device=q.device)
    same=inverse[i]==inverse[j]
    joined=predicted[i]==predicted[j]
    split=same & ~joined
    merged=~same & joined
    terms=[]
    if bool(split.any()):
        # Signed rho supplies credit even when the production product Q is zero.
        penalties=(1-rho[:,i[split],j[split]]).square().mean(0)
        terms.append(_group_mean(penalties,inverse[i[split]],classes))
    if bool(merged.any()):
        penalties=q[i[merged],j[merged]].square()
        a,b=inverse[i[merged]],inverse[j[merged]]
        pair=torch.minimum(a,b)*classes+torch.maximum(a,b)
        terms.append(_group_mean(penalties,pair,classes*classes))
    return torch.stack(terms).mean() if terms else q.sum()*0+rho.sum()*0


def loss_for_image(q,rho,labels,predicted):
    return _vectorized_class_loss(q,labels,temperature=.1)+partition_error_loss(q,rho,labels,predicted)
