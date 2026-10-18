"""Object-readout descriptors from emitted spikes/membranes only."""
import numpy as np
import torch
from collaborative_test.SW_0207_spike_foreground_classifier.classifier import node_features

def anchor_indices(grid=64,side=8,*,device='cpu'):
    if type(grid) is not int or type(side) is not int or side<1 or grid<side or grid%side:raise ValueError('nested positive anchor grid required')
    positions=torch.arange(side,device=device)*(grid//side)+(grid//side)//2
    return (positions[:,None]*grid+positions[None,:]).reshape(-1)

def descriptors(trace,rho,*,grid=64,side=8,settle=512):
    stats=node_features(trace,settle=settle)
    if stats.shape!=(1,grid*grid,12) or rho.shape!=(1,4,grid*grid,grid*grid):raise ValueError('native component/grid geometry mismatch')
    if stats.device!=rho.device or not torch.isfinite(rho).all():raise ValueError('finite same-device native rho required')
    anchors=anchor_indices(grid,side,device=rho.device)
    relations=rho[0].index_select(2,anchors).permute(1,0,2).reshape(grid*grid,4*side*side)
    values=torch.cat((relations,stats[0]),dim=1).transpose(0,1).reshape(4*side*side+12,grid,grid).contiguous().float()
    if not torch.isfinite(values).all():raise ValueError('nonfinite emitted-history descriptors')
    return values
