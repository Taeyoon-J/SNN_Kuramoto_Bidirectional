import numpy as np
import torch
NODES=4096

def encode64(encoder,patcher,mean,std,clip,rgb):
    if rgb.dtype!=torch.uint8 or rgb.ndim!=4 or tuple(rgb.shape[1:])!=(128,128,3):raise ValueError('native128 uint8 RGB required')
    if not np.isfinite(float(clip)) or clip<=0:raise ValueError('invalid feature clip')
    device=next(encoder.parameters()).device
    feature=encoder(rgb.to(device).permute(0,3,1,2).float()/255.)
    if feature.ndim!=4 or feature.shape[:2]!=(len(rgb),8):raise ValueError('eight feature channels required')
    mean=torch.as_tensor(mean,device=device,dtype=feature.dtype)
    std=torch.as_tensor(std,device=device,dtype=feature.dtype)
    if mean.shape!=std.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<=0).any():raise ValueError('invalid source statistics')
    gamma=patcher(((feature-mean)/std).clamp(-clip,clip))
    if gamma.shape!=(len(rgb),8,NODES) or not torch.isfinite(gamma).all():raise ValueError('native64 gamma geometry/finiteness')
    return gamma
