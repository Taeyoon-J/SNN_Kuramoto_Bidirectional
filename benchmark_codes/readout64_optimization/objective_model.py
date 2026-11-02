"""R3 reconstruction controls: require grouping to explain color and fine edges."""
import torch
from torch.nn import functional as F
from model import FineReadout
OBJECTIVES={
 'flat_rgb':dict(no_xy=True,edge_weight=0.),
 'flat_edge_rgb':dict(no_xy=True,edge_weight=4.),
 'spatial_edge_rgb':dict(no_xy=False,edge_weight=4.),
}
class ObjectiveReadout(FineReadout):
    def __init__(self,original,arm):
        super().__init__(original,'local_rgb');self.arm=arm
        self.objective_config=OBJECTIVES[arm]

    def forward(self,row):
        logits,coarse,prior=self.routing(row);p=(logits/.5).softmax(0)
        with torch.no_grad():content=self.content_encoder(coarse)
        latent=(p@content)/p.sum(1,keepdim=True).clamp_min(1e-6)
        xy=torch.zeros_like(self.xy) if self.objective_config['no_xy'] else self.xy
        decoded=self.decoder(torch.cat([latent[:,None].expand(-1,4096,-1),xy[None].expand(len(p),-1,-1)],-1))
        rgb=(p[:,:,None]*decoded).sum(0);errors=(rgb-row['rgb']).square().mean(-1)
        alpha=self.objective_config['edge_weight']
        if alpha:
            target=row['rgb'].reshape(64,64,3).permute(2,0,1)[None]
            smooth=F.avg_pool2d(F.pad(target,(3,3,3,3),mode='replicate'),7,stride=1)
            contrast=(target-smooth).square().sum(1).sqrt().flatten().detach()
            weights=1+alpha*(contrast/contrast.mean().clamp_min(1e-4)).clamp_max(10.)
            weights=weights/weights.mean()
        else:weights=torch.ones_like(errors)
        loss=(errors*weights).mean()
        return loss,dict(mse=errors.mean(),weighted_mse=loss),logits
