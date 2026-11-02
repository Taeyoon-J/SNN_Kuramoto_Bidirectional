"""SW0259: unchanged252 head plus shared bottleneck RGB decoder; no target masks."""
import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

HEAD_PATH=Path(__file__).resolve().parents[1]/'vendor/neutral_readout/head.py'
spec=importlib.util.spec_from_file_location('sw259_neutral_head',HEAD_PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
NeutralRegionHead=module.NeutralRegionHead


class ReconstructionReadout(nn.Module):
    def __init__(self,seed):
        super().__init__()
        self.head=NeutralRegionHead(seed=259000+seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(259100+seed)
            self.decoder=nn.Sequential(nn.Linear(34,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,3),nn.Sigmoid())
        axis=(torch.arange(32,dtype=torch.float32)+.5)/16.-1
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        self.register_buffer('xy',torch.stack([xx.flatten(),yy.flatten()],-1))

    def forward(self,features,regions):
        logits=self.head(features,regions)
        p=logits.softmax(0)
        z=self.head.embedding(features)
        mass=p.sum(1,keepdim=True).clamp_min(1e-6)
        content=(p@z)/mass
        decoded=self.decoder(torch.cat([content[:,None,:].expand(-1,1024,-1),self.xy[None].expand(len(p),-1,-1)],-1))
        rgb=(p[:,:,None]*decoded).sum(0)
        # Fixed activities define an optional self-supervised group-consistency loss.
        fixed=F.normalize((features-features.mean(0))/features.std(0,unbiased=False).clamp_min(.05),dim=-1)
        centers=(p@fixed)/mass
        coherent=F.mse_loss(p.T@centers,fixed)
        entropy=-(p*p.clamp_min(1e-8).log()).sum(0).mean()
        return rgb,dict(probability=p,coherence=coherent,entropy=entropy,
                        effective_groups=(p.sum(1)>1).sum())

    @torch.no_grad()
    def predict(self,features,regions):
        return self.head.predict(features,regions)


ARMS={
    'rgb':dict(coherence=0.,entropy=0.),
    'rgb_activity':dict(coherence=.01,entropy=0.),
    'rgb_activity_sharp':dict(coherence=.01,entropy=.001),
}


def objective(rgb,target,details,arm):
    weights=ARMS[arm]
    mse=F.mse_loss(rgb,target)
    return mse+weights['coherence']*details['coherence']+weights['entropy']*details['entropy'],mse
