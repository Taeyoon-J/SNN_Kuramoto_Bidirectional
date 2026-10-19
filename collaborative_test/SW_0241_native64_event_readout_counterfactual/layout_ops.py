"""Reproduce rollout storage and change gate values without changing spike strides."""
import numpy as np
import torch


def restore_native_spikes(values,device):
    x=torch.from_numpy(np.ascontiguousarray(values)).to(device)
    if x.ndim!=4 or x.shape[1]!=4 or x.dtype!=torch.float32:
        raise ValueError('native float32 [B,4,N,T] components required')
    b,c,n,t=x.shape
    frames=x.reshape(b*c,n,t).permute(2,0,1).contiguous()
    return frames.permute(1,2,0).reshape(b,c,n,t)


def matched_gate_only_trace(trace):
    s=trace['component_spikes'];g=trace['component_gates']
    if s.ndim!=4 or s.shape[1]!=4 or g.shape!=s.shape or g.dtype!=s.dtype or g.device!=s.device:
        raise ValueError('matching native component spikes and gates required')
    if not bool(torch.isfinite(g).all()) or bool(((g<0)|(g>1)).any()):
        raise ValueError('finite gates in [0,1] required')
    components=torch.empty_like(s,memory_format=torch.preserve_format)
    if components.stride()!=s.stride():raise ValueError('native spike storage is not reproducible')
    components.copy_(g)
    return dict(trace,component_spikes=components,spikes=components.mean(dim=1))
