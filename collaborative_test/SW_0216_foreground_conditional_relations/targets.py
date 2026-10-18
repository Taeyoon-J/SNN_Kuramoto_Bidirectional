"""TRAIN same-object relation conditional on true foreground, as in FG-ARI."""
import numpy as np


def foreground_targets(source_mask,gt,ids,pairs):
    source=np.asarray(source_mask);truth=np.asarray(gt)
    if truth.shape!=source.shape or truth.dtype.kind not in 'iu' or (truth<0).any():raise ValueError('matching nonnegative TRAIN labels required')
    objects=np.unique(truth[truth>0]);hist=np.zeros((len(ids),len(objects)),np.float64)
    for i,label in enumerate(ids):
        mask=(source==label)&(truth>0);count=int(mask.sum())
        if count:
            for j,obj in enumerate(objects):hist[i,j]=np.count_nonzero(mask&(truth==obj))/count
    a,b=np.asarray(pairs,dtype=np.int64).reshape(-1,2).T
    return np.einsum('ij,ij->i',hist[a],hist[b]).clip(0,1).astype(np.float32)
