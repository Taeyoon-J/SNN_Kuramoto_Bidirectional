"""One spectral split sweep and warm greedy remerge of the same signed energy."""
import heapq
import numpy as np


def neighbors(index,height,width):
    row,col=divmod(int(index),width)
    if row:yield index-width
    if row+1<height:yield index+width
    if col:yield index-1
    if col+1<width:yield index+1


def connected(indices,shape):
    remaining=set(map(int,indices));groups=[]
    while remaining:
        first=min(remaining);remaining.remove(first);stack=[first];group=[]
        while stack:
            node=stack.pop();group.append(node)
            for other in neighbors(node,*shape):
                if other in remaining:remaining.remove(other);stack.append(other)
        groups.append(np.asarray(sorted(group),np.int64))
    return groups


def groups_of(labels):
    flat=labels.reshape(-1)
    return sorted([np.flatnonzero(flat==i) for i in np.unique(flat[flat>0])],key=lambda x:int(x[0]))


def objective(energy,groups):return float(sum(float(energy[np.ix_(g,g)].sum())*.5 for g in groups))


def labels_of(groups,shape):
    out=np.zeros(np.prod(shape),np.int64)
    for label,group in enumerate(sorted(groups,key=lambda g:int(g.min())),1):out[group]=label
    return out.reshape(shape)


def _remerge(energy,groups,shape,tolerance):
    if len(groups)<2:return groups,0
    # Each source-FG row/column is aggregated once, retaining full-N energy values.
    summed=np.stack([energy[g].sum(0) for g in groups]);cross=np.stack([summed[:,g].sum(1) for g in groups],axis=1)
    np.fill_diagonal(cross,0.);n=len(groups);owner=np.full(energy.shape[0],-1,np.int64)
    members={i:set(map(int,g)) for i,g in enumerate(groups)};adj={i:set() for i in members}
    for i,g in enumerate(groups):owner[g]=i
    for i,g in enumerate(groups):
        for node in g:
            for other in neighbors(node,*shape):
                j=int(owner[other])
                if j>=0 and j!=i:adj[i].add(j)
    active=set(members);version=[0]*n;heap=[];merges=0
    def push(i,j):
        if i>j:i,j=j,i
        if cross[i,j]>tolerance:heapq.heappush(heap,(-float(cross[i,j]),i,j,version[i],version[j]))
    for i in sorted(active):
        for j in sorted(adj[i]):
            if i<j:push(i,j)
    while heap:
        _,i,j,vi,vj=heapq.heappop(heap)
        if i not in active or j not in active or version[i]!=vi or version[j]!=vj or j not in adj[i] or cross[i,j]<=tolerance:continue
        touching=(adj[i]|adj[j])-{i,j};cross[i,:]+=cross[j,:];cross[:,i]+=cross[:,j];cross[i,i]=0.;cross[j,:]=0.;cross[:,j]=0.
        members[i].update(members.pop(j));active.remove(j);version[i]+=1;version[j]+=1;adj[i]=touching;adj.pop(j)
        for other in touching:adj[other].discard(i);adj[other].discard(j);adj[other].add(i)
        for other in sorted(touching):push(i,other)
        merges+=1
    return [np.asarray(sorted(members[i]),np.int64) for i in sorted(active)],merges


def refine(energy,source_labels):
    source=np.asarray(source_labels);e=np.asarray(energy,dtype=np.float64)
    if source.ndim!=2 or min(source.shape)<1 or source.dtype.kind not in 'iu' or (source<0).any():raise ValueError('nonnegative integer source grid required')
    if e.shape!=(source.size,source.size) or not np.isfinite(e).all() or not np.allclose(e,e.T,rtol=0,atol=1e-12) or np.any(np.diag(e)!=0):raise ValueError('finite symmetric full-grid energy with zero diagonal required')
    original=groups_of(source);foreground=np.flatnonzero(source.reshape(-1)>0)
    if any(len(connected(g,source.shape))!=1 for g in original):raise ValueError('source group is not4-connected')
    tolerance=64*np.finfo(np.float64).eps*float(np.abs(e[np.ix_(foreground,foreground)]).sum())
    before=objective(e,original);split=[];records=[];accepted=0
    for group in original:
        block=e[np.ix_(group,group)];base=float(block.sum())*.5;tol=64*np.finfo(np.float64).eps*float(np.abs(block).sum())
        if len(group)<2:split.append(group);continue
        laplacian=np.diag(block.sum(1))-block;values,vectors=np.linalg.eigh(laplacian)
        record=dict(first_patch=int(group[0]),nodes=len(group),minimum_eigenvalue=float(values[0]),accepted=False)
        if values[0]>=-tol:split.append(group);records.append(record);continue
        vector=vectors[:,0];order=np.argsort(vector,kind='stable');ordered=block[np.ix_(order,order)]
        increments=ordered.sum(1)-2*np.tril(ordered,k=-1).sum(1);cuts=np.cumsum(increments)[:-1]
        prefix=int(np.argmin(cuts))+1
        candidates=[vector<0,np.isin(np.arange(len(group)),order[:prefix])];best=[group];best_value=base
        for mask in candidates:
            if not mask.any() or mask.all():continue
            pieces=connected(group[mask],source.shape)+connected(group[~mask],source.shape)
            value=objective(e,pieces)
            if value>best_value+tol:best=pieces;best_value=value
        if len(best)>1:accepted+=1;record.update(accepted=True,new_groups=len(best),energy_gain=best_value-base)
        split.extend(best);records.append(record)
    split.sort(key=lambda g:int(g[0]));split_value=objective(e,split)
    merged,merges=_remerge(e,split,source.shape,tolerance);after=objective(e,merged)
    if split_value<before-tolerance or after<split_value-tolerance:raise AssertionError('refinement decreased original signed objective')
    split_labels=source.copy() if accepted==0 else labels_of(split,source.shape)
    final_labels=source.copy() if accepted==0 and merges==0 else labels_of(merged,source.shape)
    for labels in (split_labels,final_labels):
        if not np.array_equal(labels>0,source>0) or any(len(connected(g,source.shape))!=1 for g in groups_of(labels)):raise AssertionError('refinement violated source foreground/connectivity')
    return dict(split=split_labels,refined=final_labels,diagnostic=dict(original_groups=len(original),split_groups=len(split),final_groups=len(merged),
        accepted_source_splits=accepted,remerges=merges,energy_before=before,energy_split=split_value,energy_refined=after,tolerance=tolerance,proposals=records))
