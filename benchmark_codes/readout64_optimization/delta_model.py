"""SW0260R2: exact baseline hard routing at initialization, learned bounded changes."""
import torch
from torch.nn import functional as F
from model import FineReadout,expand32

DELTA_ARMS={
 'delta_prior1':dict(radius=1,tv=0.,spike=0.,anchor=0.,prior=1.,trust=1.5),
 'delta_prior05':dict(radius=1,tv=0.,spike=0.,anchor=0.,prior=.5,trust=1.),
 'delta_spike_trust':dict(radius=1,tv=0.,spike=.003,anchor=.001,prior=.5,trust=1.),
}

class DeltaReadout(FineReadout):
    def __init__(self,original,arm):
        super().__init__(original,'local_rgb');self.arm=arm;self.config=DELTA_ARMS[arm]

    def routing(self,row):
        current,coarse,prior=super().routing(row)
        allowed=current>-9999
        calibrated_current=(current-prior)/2.
        with torch.no_grad():
            masks=expand32(row['masks'].T).T.float()
            reference=F.normalize(self.content_encoder(coarse),dim=-1)
            centers=F.normalize(masks@reference/masks.sum(1,keepdim=True).clamp_min(1),dim=-1)
            sim=centers@reference.T
            weight=allowed.float();mean=(sim*weight).sum(0)/weight.sum(0).clamp_min(1)
            std=(((sim-mean).square()*weight).sum(0)/weight.sum(0).clamp_min(1)).clamp_min(.0025).sqrt()
            reference_calibrated=((sim-mean)/std).tanh()
        delta=self.config['trust']*(2.*(calibrated_current-reference_calibrated)).tanh()
        logits=(self.config['prior']*prior+delta).masked_fill(~allowed,-1e4)
        return logits,coarse,prior
