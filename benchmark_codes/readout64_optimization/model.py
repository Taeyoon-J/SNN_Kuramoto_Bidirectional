"""SW0260: preserve32-core activities, compute fine-image routing at64 resolution."""
import copy
import torch
from torch import nn
from torch.nn import functional as F

ARMS={
 'local_rgb':dict(radius=1,tv=0.,spike=0.,anchor=0.),
 'local_rgb_tv':dict(radius=1,tv=.01,spike=0.,anchor=0.),
 'local_rgb_spike':dict(radius=1,tv=0.,spike=.01,anchor=0.),
 'local_rgb_trust':dict(radius=1,tv=0.,spike=.01,anchor=.002),
 'local_rgb_r2':dict(radius=2,tv=.01,spike=.01,anchor=.002),
}


def expand32(x):
    # x[N32,C] -> x[N64,C], preserving the actual coarse activity values.
    return x.reshape(32,32,-1).repeat_interleave(2,0).repeat_interleave(2,1).reshape(4096,-1)


class FineReadout(nn.Module):
    def __init__(self,original,arm):
        super().__init__();self.arm=arm;self.config=ARMS[arm]
        self.head=copy.deepcopy(original.head)
        self.head.prior_scale.requires_grad_(False);self.head.residual_scale.requires_grad_(False)
        # Frozen content eliminates the direct embedding-to-RGB gradient bypass.
        self.content_encoder=copy.deepcopy(original.head.embedding).requires_grad_(False)
        self.decoder=copy.deepcopy(original.decoder)
        self.fine=nn.Linear(8,32,bias=False);nn.init.zeros_(self.fine.weight)
        axis=(torch.arange(64,dtype=torch.float32)+.5)/32.-1
        yy,xx=torch.meshgrid(axis,axis,indexing='ij');self.register_buffer('xy',torch.stack([xx.flatten(),yy.flatten()],-1))

    def routing(self,row):
        coarse=expand32(row['features'])
        masks=expand32(row['masks'].T).T.bool();prior=expand32(row['prior'].T).T
        z=F.normalize(self.head.embedding(coarse)+self.fine(row['gamma']),dim=-1)
        prototypes=F.normalize(masks.float()@z/masks.sum(1,keepdim=True).clamp_min(1),dim=-1)
        similarity=prototypes@z.T
        radius=self.config['radius'];allowed=F.max_pool2d(masks.float().reshape(-1,1,64,64),2*radius+1,stride=1,padding=radius).flatten(1).bool()
        weights=allowed.float();mean=(similarity*weights).sum(0)/weights.sum(0).clamp_min(1)
        std=(((similarity-mean).square()*weights).sum(0)/weights.sum(0).clamp_min(1)).clamp_min(.0025).sqrt()
        calibrated=((similarity-mean)/std).tanh()
        logits=(prior+2.*calibrated).masked_fill(~allowed,-1e4)
        return logits,coarse,prior

    def forward(self,row):
        logits,coarse,prior=self.routing(row);p=(logits/.5).softmax(0)
        with torch.no_grad():content=self.content_encoder(coarse)
        latent=(p@content)/p.sum(1,keepdim=True).clamp_min(1e-6)
        decoded=self.decoder(torch.cat([latent[:,None].expand(-1,4096,-1),self.xy[None].expand(len(p),-1,-1)],-1))
        rgb=(p[:,:,None]*decoded).sum(0)
        mse=F.mse_loss(rgb,row['rgb'])
        grid=p.reshape(-1,64,64);target=row['rgb'].reshape(64,64,3)
        tv=[];pair=[]
        temporal=expand32(row['features'][:,:128]).reshape(64,64,4,32)
        temporal=F.normalize(temporal-temporal.mean(-1,keepdim=True),dim=-1)
        for a,b,ca,cb,ta,tb in [
            (grid[:,:,1:],grid[:,:,:-1],target[:,1:],target[:,:-1],temporal[:,1:],temporal[:,:-1]),
            (grid[:,1:,:],grid[:,:-1,:],target[1:,:],target[:-1,:],temporal[1:,:],temporal[:-1,:])]:
            edge=((ca-cb).square().sum(-1)/.02).neg().exp().detach()
            tv.append(((a-b).abs().sum(0)*edge).mean())
            affinity=(ta*tb).sum(-1).clamp_min(0).prod(-1).detach()
            pair.append(F.mse_loss((a*b).sum(0),affinity))
        tv=sum(tv)/2;pair=sum(pair)/2
        pseudo=prior.argmax(0)  # self-generated proposal labels, never dataset masks.
        anchor=F.cross_entropy(logits.T/.5,pseudo)
        loss=mse+self.config['tv']*tv+self.config['spike']*pair+self.config['anchor']*anchor
        return loss,dict(mse=mse,tv=tv,spike=pair,anchor=anchor),logits

    @torch.no_grad()
    def predict(self,row):return (self.routing(row)[0].argmax(0)+1).reshape(64,64)
