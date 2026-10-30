"""SW0252: exchangeable region groups, including background, with no deletion."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def candidates(features, regions):
    labels=torch.as_tensor(regions,device=features.device,dtype=torch.long).flatten()
    if labels.numel()!=len(features) or bool((labels<0).any()):
        raise ValueError('aligned complete region partition required')
    masks=labels[None]==torch.unique(labels)[:,None]
    assert bool((masks.sum(0)==1).all())  # Include ID 0 as an ordinary region too.
    original=len(masks)
    with torch.no_grad():
        z=features.detach().float()
        z=F.normalize((z-z.mean(0))/z.std(0,unbiased=False).clamp_min(.05),dim=-1)
        centers=[];nearest=torch.full((len(z),),float('inf'),device=z.device)
        for _ in range(min(12,len(z))):
            if centers:nearest[torch.tensor(centers,device=z.device)]=-1
            i=int(nearest.argmax()) if centers else int((z-z.mean(0)).square().sum(1).argmax())
            centers.append(i);nearest=torch.minimum(nearest,(z-z[i]).square().sum(1))
        distances=torch.stack([(z-z[i]).square().sum(1) for i in centers])
        assignment=distances.argmin(0)
        for j,i in enumerate(centers):assignment[i]=j
        extra=assignment[None]==torch.arange(len(centers),device=z.device)[:,None]
    all_masks=torch.cat([masks,extra])
    prior=torch.zeros(all_masks.shape,device=z.device)
    prior[:original]=masks.float()
    return all_masks,prior


class NeutralRegionHead(nn.Module):
    def __init__(self,seed=251000):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.embedding=nn.Sequential(nn.LayerNorm(144),nn.Linear(144,64),nn.SiLU(),nn.Linear(64,32))
        self.residual_scale=nn.Parameter(torch.zeros(()))
        self.prior_scale=nn.Parameter(torch.tensor(4.))

    def forward(self,features,regions):
        if features.ndim!=2 or features.shape[1]!=144 or not bool(torch.isfinite(features).all()):
            raise ValueError('finite actual spike/membrane features required')
        masks,prior=candidates(features,regions)
        z=F.normalize(self.embedding(features),dim=-1)
        prototypes=F.normalize(masks.float()@z/masks.sum(1,keepdim=True).clamp_min(1),dim=-1)
        return self.prior_scale*prior+self.residual_scale*(prototypes@z.T)

    @torch.no_grad()
    def predict(self,features,regions):
        # Every output is an ordinary positive region ID. No background designation.
        return (self(features,regions).argmax(0)+1).reshape(np.shape(regions)).cpu().numpy().astype(np.int64)


def objective(logits,labels):
    """Match ALL truth regions, including ID0, to interchangeable predictions."""
    from scipy.optimize import linear_sum_assignment
    truth=torch.as_tensor(labels,device=logits.device,dtype=torch.long).flatten()
    classes=torch.unique(truth)
    if len(classes)>len(logits):raise ValueError('insufficient inference proposals; GT cannot add proposals')
    probability=logits.softmax(0)
    masks=(truth[None]==classes[:,None]).float()
    dice=(2*(probability@masks.T)+1e-6)/(probability.sum(1)[:,None]+masks.sum(1)[None]+1e-6)
    rows,cols=linear_sum_assignment(-dice.detach().cpu().numpy())
    assert len(cols)==len(classes)
    target=torch.empty_like(truth)
    for row,col in zip(rows,cols):target[truth==classes[col]]=int(row)
    logp=logits.log_softmax(0)
    ce=torch.stack([-logp[target[truth==k],(truth==k).nonzero().flatten()].mean() for k in classes]).mean()
    mask_dice=1-dice[torch.as_tensor(rows,device=logits.device),torch.as_tensor(cols,device=logits.device)].mean()
    # No background BCE, foreground-only pairing, largest-region rule, or BG anchor.
    loss=ce+mask_dice
    if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite neutral mask loss')
    return loss
