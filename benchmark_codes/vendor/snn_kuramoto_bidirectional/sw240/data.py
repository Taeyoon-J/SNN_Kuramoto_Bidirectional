"""Canonical128 RGB, native64 modal labels and reserved evaluation exclusion."""
import numpy as np


def train_ids(count=35000):
    pool=np.r_[np.arange(1000),np.arange(1640,70640)].astype(np.int64)
    if count==70000:
        return pool
    if count!=35000:
        raise ValueError('35,000 or 70,000 TRAIN images required')
    return np.sort(np.random.RandomState(24935000).choice(pool,count,replace=False))


def modal_labels(pixels):
    pixels=np.asarray(pixels)
    if pixels.ndim==4 and pixels.shape[-1]==1:
        pixels=pixels[...,0]
    if pixels.ndim!=3 or pixels.shape[1:]!=(128,128) or pixels.dtype.kind not in 'iu' or pixels.min()<0:
        raise ValueError('nonnegative integer [B,128,128] masks required')
    blocks=pixels.reshape(len(pixels),64,2,64,2).transpose(0,1,3,2,4).reshape(len(pixels),64,64,4)
    ordered=np.sort(blocks,axis=-1)
    counts=(ordered[..., :,None]==ordered[...,None,:]).sum(-1)
    return np.take_along_axis(ordered,counts.argmax(-1)[...,None],axis=-1)[...,0].astype(np.int64)


def read_batch(stream,ids):
    ids=np.asarray(ids)
    if (ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=len(ids)
            or np.any(ids<0) or np.any(ids>=70640) or np.any((ids>=1000)&(ids<1640))):
        raise ValueError('unique TRAIN IDs excluding 1000..1639 required')
    order=np.argsort(ids);inverse=np.argsort(order)
    rgb=np.asarray(stream['image'][ids[order]])[inverse]
    if rgb.ndim!=4 or rgb.shape[1:]!=(128,128,3):
        raise ValueError('native128 RGB required')
    if rgb.dtype!=np.uint8:
        if rgb.dtype.kind!='f' or not np.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>1:
            raise ValueError('uint8 or normalized finite RGB required')
        rgb=np.rint(rgb.astype(np.float32)*255).astype(np.uint8)
    return rgb,modal_labels(np.asarray(stream['mask'][ids[order]]))[inverse]
