"""Align valid-convolution feature centers with the unchanged native2px cells."""
import numpy as np
import torch
import torch.nn.functional as F


def geometry():
    start=np.floor(np.arange(64)*126/64).astype(int);end=np.ceil((np.arange(64)+1)*126/64).astype(int)
    feature=torch.arange(1,127,dtype=torch.float32).view(1,1,1,126).expand(1,1,126,126)
    target=2*np.arange(64)+.5
    old=F.adaptive_avg_pool2d(feature,(64,64))[0,0,32].numpy()
    aligned=F.adaptive_avg_pool2d(F.pad(feature,(1,1,1,1),mode='replicate'),(64,64))[0,0,32].numpy()
    return dict(feature_shape=[126,126],cell_grid=[64,64],legacy_window_width_counts={str(k):int((end-start==k).sum()) for k in np.unique(end-start)},
        legacy_overlapping_adjacent_cells_per_axis=int((end[:-1]>start[1:]).sum()),legacy_mean_center_abs_error_pixels=float(np.abs(old-target).mean()),
        aligned_mean_center_abs_error_pixels=float(np.abs(aligned-target).mean()),aligned_interior_max_center_error_pixels=float(np.abs(aligned[1:-1]-target[1:-1]).max()),
        border_policy='replicate one existing feature row/column; no RGB or GT resize/crop')


def encode(found,rgb):
    if rgb.dtype!=torch.uint8 or rgb.ndim!=4 or tuple(rgb.shape[1:])!=(128,128,3):raise ValueError('unchanged canonical128 uint8 RGB required')
    device=next(found.encoder.parameters()).device
    feature=found.encoder(rgb.to(device).permute(0,3,1,2).float()/255.)
    if feature.shape!=(len(rgb),8,126,126):raise ValueError('actual valid-convolution feature geometry required')
    mean=torch.as_tensor(found.feature_mean,device=device,dtype=feature.dtype);std=torch.as_tensor(found.feature_std,device=device,dtype=feature.dtype)
    clip=float(found.feature_clip)
    if mean.shape!=std.shape or not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()) or not bool((std>0).all()) or not np.isfinite(clip) or clip<=0:raise ValueError('source statistics must remain finite and fixed')
    normalized=((feature-mean)/std).clamp(-clip,clip)
    # Interior feature values remain exact. Only the pooling grid changes.
    gamma=found.patcher(F.pad(normalized,(1,1,1,1),mode='replicate'))
    if gamma.shape!=(len(rgb),8,4096) or not bool(torch.isfinite(gamma).all()):raise ValueError('native64 gamma geometry/finiteness')
    return gamma
