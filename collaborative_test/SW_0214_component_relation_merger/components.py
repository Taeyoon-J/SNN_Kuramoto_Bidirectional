"""Canonical source components and symmetric spike-derived pair features."""
import numpy as np


def component_pairs(features,source_mask,channel_mean,channel_scale):
    x=np.asarray(features);y=np.asarray(source_mask)
    if y.ndim!=2 or y.shape[0]!=y.shape[1] or y.dtype.kind not in 'iu' or (y<0).any() or x.shape!=(268,*y.shape) or not np.isfinite(x).all():raise ValueError('finite268-channel source component grid required')
    z=((x-np.asarray(channel_mean).reshape(268,1,1))/np.asarray(channel_scale).reshape(268,1,1)).clip(-5,5)
    ids=sorted(np.unique(y[y>0]).tolist(),key=lambda i:int(np.flatnonzero(y==i)[0]))
    desc=[];geometry=[];grid=len(y)
    for label in ids:
        mask=y==label;yy,xx=np.nonzero(mask);height=int(yy.max()-yy.min()+1);width=int(xx.max()-xx.min()+1)
        desc.append(z[:,mask].mean(1));geometry.append([yy.mean()/max(grid-1,1),xx.mean()/max(grid-1,1),height/grid,width/grid,len(yy)/(grid*grid),len(yy)/(height*width)])
    pairs=np.asarray([(i,j) for i in range(len(ids)) for j in range(i+1,len(ids))],dtype=np.int64).reshape(-1,2)
    if not len(pairs):return np.asarray(ids,np.int64),pairs,np.empty((0,548),np.float32)
    desc=np.asarray(desc);geometry=np.asarray(geometry);a,b=pairs.T
    values=np.concatenate((abs(desc[a]-desc[b]),desc[a]*desc[b],abs(geometry[a]-geometry[b]),geometry[a]*geometry[b]),axis=1).astype(np.float32)
    if values.shape!=(len(pairs),548) or not np.isfinite(values).all():raise ValueError('invalid component relation features')
    return np.asarray(ids,np.int64),pairs,values


def soft_targets(source_mask,gt,ids,pairs):
    y=np.asarray(source_mask);g=np.asarray(gt)
    if g.shape!=y.shape or g.dtype.kind not in 'iu' or (g<0).any():raise ValueError('TRAIN-only matching instance labels required')
    objects=np.unique(g[g>0]);hist=np.zeros((len(ids),len(objects)),np.float64)
    for i,label in enumerate(ids):
        mask=y==label
        for j,obj in enumerate(objects):hist[i,j]=np.count_nonzero(mask&(g==obj))/max(mask.sum(),1)
    a,b=pairs.T
    return np.einsum('ij,ij->i',hist[a],hist[b]).clip(0,1).astype(np.float32)


def merge(source_mask,ids,pairs,logits):
    source=np.asarray(source_mask);values=np.asarray(logits).reshape(-1)
    if values.shape!=(len(pairs),) or not np.isfinite(values).all():raise ValueError('finite pair logits required')
    matrix=np.zeros((len(ids),len(ids)),np.float64)
    if len(pairs):matrix[pairs[:,0],pairs[:,1]]=values;matrix[pairs[:,1],pairs[:,0]]=values
    groups=[[i] for i in range(len(ids))]
    while True:
        best=None
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                score=float(matrix[np.ix_(groups[i],groups[j])].min())
                if score>0 and (best is None or score>best[0]):best=(score,i,j)
        if best is None:break
        _,i,j=best;groups[i]=sorted(groups[i]+groups[j]);del groups[j]
    out=source.copy().astype(np.int64)
    for group in groups:
        for index in group:out[source==ids[index]]=ids[group[0]]
    if not np.array_equal(out>0,source>0):raise AssertionError('merger changed source foreground')
    return out,dict(original_components=len(ids),predicted_objects=len(groups),merges=len(ids)-len(groups))
