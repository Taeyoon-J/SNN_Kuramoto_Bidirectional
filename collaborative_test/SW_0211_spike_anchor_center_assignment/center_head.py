"""Learned center voting with immutable spike-derived foreground support."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CHANNELS=268

class CenterHead(nn.Module):
    def __init__(self,channels=CHANNELS,width=32):
        super().__init__()
        if width%8:raise ValueError('eight normalization groups required')
        self.register_buffer('mean',torch.zeros(1,channels,1,1));self.register_buffer('scale',torch.ones(1,channels,1,1))
        layers=[nn.Conv2d(channels,width,1),nn.GroupNorm(8,width),nn.SiLU()]
        for dilation in (1,2,4,8):layers += [nn.Conv2d(width,width,3,padding=dilation,dilation=dilation),nn.GroupNorm(8,width),nn.SiLU()]
        self.context=nn.Sequential(*layers);self.output=nn.Conv2d(width,4,1)
        with torch.no_grad():
            self.output.weight.normal_(0,.001);self.output.bias.zero_()
            self.output.bias[1]=math.log(.01/.99);self.output.weight[2:].zero_()
    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=self.mean.shape[1]:raise ValueError('BCHW spike descriptors required')
        output=self.output(self.context(((x-self.mean)/self.scale).clamp(-5,5)))
        return dict(foreground=output[:,0],heat=output[:,1],offset=output[:,2:])

def normalization(features):
    x=np.asarray(features)
    if x.dtype!=np.float32 or x.ndim!=4 or len(x)<1 or not np.isfinite(x).all():raise ValueError('finite TRAIN BCHW float32 required')
    mean=x.mean((0,2,3),dtype=np.float64).astype(np.float32)
    scale=np.sqrt(x.var((0,2,3),dtype=np.float64)).clip(1e-4).astype(np.float32)
    return mean.reshape(1,-1,1,1),scale.reshape(1,-1,1,1)

def targets(labels,*,sigma=1.5):
    y=np.asarray(labels)
    if y.ndim!=2 or y.shape[0]!=y.shape[1] or y.dtype.kind not in 'iu' or (y<0).any() or not np.isfinite(sigma) or sigma<=0:raise ValueError('nonnegative square TRAIN instance grid required')
    grid=len(y);objects=np.unique(y[y>0]);classes=np.unique(y)
    if not len(objects):raise ValueError('TRAIN image needs foreground objects')
    yy,xx=np.mgrid[:grid,:grid];heat=np.zeros(y.shape,np.float32);offset=np.zeros((2,grid,grid),np.float32)
    fg_weights=np.zeros(y.shape,np.float32);offset_weights=np.zeros(y.shape,np.float32);centers=[]
    for label in classes:
        mask=y==label;fg_weights[mask]=1/(len(classes)*int(mask.sum()))
    for obj in objects:
        mask=y==obj;positions=np.argwhere(mask);centroid=positions.mean(0)
        representative=positions[np.argmin(((positions-centroid)**2).sum(1))];cy,cx=map(int,representative)
        centers.append([cy,cx]);heat=np.maximum(heat,np.exp(-((yy-cy)**2+(xx-cx)**2)/(2*sigma*sigma)).astype(np.float32))
        offset[0,mask]=(cx-xx[mask])/max(grid-1,1);offset[1,mask]=(cy-yy[mask])/max(grid-1,1)
        offset_weights[mask]=1/(len(objects)*int(mask.sum()))
    return dict(foreground=(y>0).astype(np.float32),foreground_weights=fg_weights,heat=heat,offset=offset,
        offset_weights=offset_weights,object_count=np.int64(len(objects)),centers=np.asarray(centers,dtype=np.int64))

def loss(output,target):
    fg=output['foreground'];heat=output['heat'];offset=output['offset'];grid=fg.shape[-1]
    if fg.shape!=target['foreground'].shape or heat.shape!=fg.shape or offset.shape!=(len(fg),2,grid,grid):raise ValueError('target/head geometry mismatch')
    foreground=(F.binary_cross_entropy_with_logits(fg,target['foreground'],reduction='none')*target['foreground_weights']).sum((1,2)).mean()
    probability=heat.sigmoid().clamp(1e-6,1-1e-6);truth=target['heat'];positive=truth==1
    focal=-torch.log(probability)*(1-probability).square()*positive
    focal-=torch.log1p(-probability)*probability.square()*(1-truth).pow(4)*(~positive)
    center=(focal.sum((1,2))/target['object_count'].clamp_min(1)).mean()
    delta=F.smooth_l1_loss(offset*(grid-1),target['offset']*(grid-1),reduction='none',beta=1).sum(1)
    displacement=(delta*target['offset_weights']).sum((1,2)).mean()
    total=foreground+center+displacement
    if not torch.isfinite(total):raise FloatingPointError('nonfinite center-voting objective')
    return total,dict(foreground=float(foreground.detach()),center=float(center.detach()),offset=float(displacement.detach()))

def decode(heat_logits,normalized_offsets,source_labels,*,threshold=.3,max_centers=10,radius=1):
    heat=np.asarray(heat_logits);offset=np.asarray(normalized_offsets);labels=np.asarray(source_labels)
    if labels.ndim!=2 or labels.shape[0]!=labels.shape[1] or labels.dtype.kind not in 'iu' or (labels<0).any():raise ValueError('nonnegative square source mask required')
    grid=len(labels)
    if heat.shape!=labels.shape or offset.shape!=(2,grid,grid) or not np.isfinite(heat).all() or not np.isfinite(offset).all():raise ValueError('finite matching center/offset fields required')
    if not 0<threshold<1 or type(max_centers) is not int or max_centers<1 or radius<0:raise ValueError('invalid fixed decoder constants')
    support=labels>0;out=np.zeros(labels.shape,np.int64)
    if not support.any():return out,dict(centers=[],fallback=False,predicted_objects=0)
    probability=1/(1+np.exp(-np.clip(heat,-80,80)))
    padded=np.pad(probability,1,constant_values=-np.inf)
    local=np.maximum.reduce([padded[y:y+grid,x:x+grid] for y in range(3) for x in range(3)])
    indices=np.flatnonzero(support & (probability>=threshold) & (probability==local))
    indices=indices[np.lexsort((indices,-probability.reshape(-1)[indices]))]
    centers=[]
    for index in indices:
        point=np.asarray(divmod(int(index),grid))
        if all(float(((point-np.asarray(c))**2).sum())>radius*radius for c in centers):centers.append(point.tolist())
        if len(centers)==max_centers:break
    fallback=not bool(centers)
    if fallback:
        available=np.flatnonzero(support);best=available[np.argmax(probability.reshape(-1)[available])];centers=[list(divmod(int(best),grid))]
    yy,xx=np.nonzero(support);votes=np.stack((yy+offset[1,yy,xx]*(grid-1),xx+offset[0,yy,xx]*(grid-1)),axis=1)
    nearest=((votes[:,None,:]-np.asarray(centers)[None,:,:])**2).sum(2).argmin(1)
    out[yy,xx]=nearest+1
    if not np.array_equal(out>0,support):raise AssertionError('center decoder changed spike foreground')
    return out,dict(centers=centers,fallback=fallback,predicted_objects=int(np.unique(out[out>0]).size))
