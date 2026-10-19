"""GT-assisted diagnostic operations; never used to predict or select a model."""
import numpy as np


def transform(prediction,target,merge=False,cover=False):
    p=np.asarray(prediction);t=np.asarray(target)
    if p.shape!=t.shape or p.ndim!=2 or p.dtype.kind not in 'iu' or t.dtype.kind not in 'iu' or (p<0).any() or (t<0).any():raise ValueError('aligned nonnegative integer masks required')
    out=p.copy()
    if merge:
        spare=int(max(p.max(),t.max()))+1
        for group in np.unique(p):
            if group==0:continue
            take=p==group;ids,counts=np.unique(t[take],return_counts=True);fg=ids!=0
            if fg.any():out[take]=ids[fg][counts[fg].argmax()]
            else:out[take]=spare;spare+=1
        if not np.array_equal(out>0,p>0):raise ValueError('merge changed foreground support')
    if cover:
        spare=int(out.max())+1
        for obj in np.unique(t):
            if obj==0:continue
            here=t==obj;ids,counts=np.unique(out[here],return_counts=True);fg=ids!=0
            if fg.any():best=ids[fg][counts[fg].argmax()]
            else:best=spare;spare+=1
            out[here & (out==0)]=best
        if not np.array_equal(out[p>0],p[p>0]) and not merge:raise ValueError('coverage changed existing labels')
    return out
