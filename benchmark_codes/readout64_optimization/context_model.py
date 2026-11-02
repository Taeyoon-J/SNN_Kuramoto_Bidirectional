"""R4: learn spatial context only in the64-grid fine readout branch, core untouched."""
import torch
from torch import nn
from objective_model import ObjectiveReadout
CONTEXT_ARMS={
 'context_rgb':dict(no_xy=False,edge_weight=0.,dilation=1),
 'context_flat_edge':dict(no_xy=True,edge_weight=4.,dilation=1),
 'context_wide_rgb':dict(no_xy=False,edge_weight=0.,dilation=2),
}
class ContextAdapter(nn.Module):
    def __init__(self,seed,dilation):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(262000+seed)
            self.net=nn.Sequential(nn.Conv2d(8,16,3,padding=dilation,dilation=dilation),nn.SiLU(),nn.Conv2d(16,32,3,padding=dilation,dilation=dilation,bias=False))
            nn.init.zeros_(self.net[-1].weight)
    @property
    def weight(self):return self.net[-1].weight
    def forward(self,gamma):
        x=gamma.reshape(64,64,8).permute(2,0,1)[None]
        return self.net(x)[0].permute(1,2,0).reshape(4096,32)

class ContextReadout(ObjectiveReadout):
    def __init__(self,original,arm):
        super().__init__(original,'flat_rgb');self.arm=arm
        self.objective_config=CONTEXT_ARMS[arm]
        self.fine=ContextAdapter(getattr(original,'_seed_hint',0),self.objective_config['dilation'])
