"""Provisional CPU components for SW0167; no data or runtime orchestration."""
from __future__ import annotations

import heapq
import numpy as np
import torch
from torch import nn

from collaborative_test.SW_0162_connected_signed_partition.connected_signed import signed_pair_energy


def solve_energy(energy, source_labels, *, grid_shape):
    """SW0164 greedy merge operating directly on a symmetric energy matrix."""
    h, w = _grid_shape(grid_shape)
    n = h * w
    e = np.asarray(energy, dtype=np.float64)
    support_labels = np.asarray(source_labels)
    if e.shape != (n, n) or support_labels.shape != (h, w):
        raise ValueError("energy/support do not match grid shape")
    if not np.isfinite(e).all() or not np.allclose(e, e.T, rtol=0, atol=1e-12):
        raise ValueError("energy must be finite and symmetric")
    if not np.issubdtype(support_labels.dtype, np.integer) or np.any(support_labels < 0):
        raise ValueError("source labels must be nonnegative integers")
    support = np.flatnonzero(support_labels.reshape(-1) > 0).astype(np.int64)
    out = np.zeros(n, dtype=np.int32)
    if support.size == 0:
        return {"labels": out.reshape(h, w), "clusters": [], "merge_count": 0,
                "merge_log": [], "foreground_support": support, "pair_energy": e.copy()}
    active = set(map(int, support))
    members = {i: {i} for i in active}
    neighbors = {i: {j for j in _neighbors(i, h, w) if j in active} for i in active}
    cross = e.copy()
    version = [0] * n
    heap = []

    def push(x, y):
        if x == y:
            return
        if x > y:
            x, y = y, x
        value = float(cross[x, y])
        if value > 0:
            heapq.heappush(heap, (-value, x, y, version[x], version[y]))

    for x in sorted(active):
        for y in sorted(neighbors[x]):
            if x < y:
                push(x, y)
    merge_log = []
    while heap:
        _, x, y, vx, vy = heapq.heappop(heap)
        if (x not in active or y not in active or version[x] != vx or version[y] != vy
                or y not in neighbors[x]):
            continue
        gain = float(cross[x, y])
        if gain <= 0:
            continue
        keep, drop = x, y
        touching = (neighbors[keep] | neighbors[drop]) - {keep, drop}
        cross[keep, :] += cross[drop, :]
        cross[:, keep] += cross[:, drop]
        cross[keep, keep] = 0.0
        cross[drop, :] = 0.0
        cross[:, drop] = 0.0
        members[keep].update(members.pop(drop))
        active.remove(drop)
        version[keep] += 1
        version[drop] += 1
        neighbors[keep] = set(touching)
        neighbors.pop(drop)
        for other in touching:
            neighbors[other].discard(keep)
            neighbors[other].discard(drop)
            neighbors[other].add(keep)
        for other in sorted(touching):
            push(keep, other)
        merge_log.append({"patch_pair": [x, y], "gain": gain})
    groups = sorted((sorted(group) for group in members.values()), key=lambda group: group[0])
    clusters = []
    for label, points in enumerate(groups, start=1):
        out[points] = label
        clusters.append({"label": label, "representative": points[0],
                         "size": len(points), "patches": points})
    if not np.array_equal(out > 0, support_labels.reshape(-1) > 0):
        raise AssertionError("induced solver changed fixed foreground support")
    return {"labels": out.reshape(h, w), "clusters": clusters,
            "merge_count": len(merge_log), "merge_log": merge_log,
            "foreground_support": support, "pair_energy": e.copy()}


def base_energy(q, source_a, gain):
    """Return the exact registered SW0162/SW0164 full-grid signed energy."""
    return signed_pair_energy(q, source_a, gain)


def _grid_shape(shape):
    if (not isinstance(shape, (tuple, list)) or len(shape) != 2
            or any(type(v) is not int or v <= 0 for v in shape)):
        raise ValueError("grid_shape must contain two positive integers")
    return int(shape[0]), int(shape[1])


def _neighbors(i, h, w):
    r, c = divmod(i, w)
    if r: yield i-w
    if r+1 < h: yield i+w
    if c: yield i-1
    if c+1 < w: yield i+1


def pair_features(component_spikes, q, source_a, gain, *, eps=1e-8):
    """Build symmetric 10-channel pair features from actual settled spikes/Q/A.

    Inputs are [4,N,T], [N,N], [N,N]. No RGB, positions, labels, or counts enter.
    """
    s = torch.as_tensor(component_spikes)
    q = torch.as_tensor(q, device=s.device, dtype=torch.float32)
    a = torch.as_tensor(source_a, device=s.device, dtype=torch.float32)
    if s.ndim != 3 or s.shape[0] != 4 or not s.is_floating_point():
        raise ValueError("component spikes must be floating [4,N,T]")
    n = s.shape[1]
    if s.shape[2] < 2 or q.shape != (n,n) or a.shape != (n,n):
        raise ValueError("spike/Q/A geometry mismatch")
    if not torch.isfinite(s).all() or not torch.isfinite(q).all() or not torch.isfinite(a).all():
        raise ValueError("features require finite source tensors")
    if torch.any(q < -1e-7) or torch.any(q > 1+1e-7) or not torch.allclose(q,q.T,atol=2e-6,rtol=0):
        raise ValueError("Q must be symmetric and lie in [0,1]")
    if not torch.allclose(a,a.T,atol=2e-6,rtol=0):
        raise ValueError("actual source A must be symmetric")
    if not np.isfinite(float(gain)) or float(gain) <= 0:
        raise ValueError("coupling gain must be finite and positive")
    # Match spike_synchrony_affinity's component loop and operation order.
    per_component=[]
    for d in range(4):
        x=s[d].float()
        x=x-x.mean(dim=-1,keepdim=True)
        x=x/x.norm(dim=-1,keepdim=True).clamp_min(eps)
        per_component.append((x@x.transpose(-1,-2)).clamp(-1,1))
    rho=torch.stack(per_component,dim=0)
    q_rebuilt=rho.clamp_min(0).prod(dim=0)
    if not torch.allclose(q,q_rebuilt,rtol=0,atol=2e-6):
        raise ValueError("Q does not reconstruct from production component-spike Pearson products")
    qsym=(q+q.T)*0.5
    qbar=qsym.clamp(1e-6,1-1e-6)
    logit=torch.log(qbar/(1-qbar))
    asym=(a+a.T)*0.5
    if torch.any(asym<0): raise ValueError("prepared source coupling A must be nonnegative")
    loga=torch.log1p(1024.0*asym/float(gain))
    eye=torch.eye(n,device=s.device,dtype=torch.bool)
    qstats=q.masked_fill(eye,0)
    denom=max(n-1,1)
    row_mean=qstats.sum(-1)/denom
    row_var=(qstats.square().sum(-1)/denom-row_mean.square()).clamp_min(0)
    row_std=row_var.sqrt()
    mean_avg=(row_mean[:,None]+row_mean[None,:])*0.5
    mean_diff=(row_mean[:,None]-row_mean[None,:]).abs()
    std_avg=(row_std[:,None]+row_std[None,:])*0.5
    std_diff=(row_std[:,None]-row_std[None,:]).abs()
    features=torch.cat((rho.permute(1,2,0),logit[...,None],loga[...,None],mean_avg[...,None],mean_diff[...,None],
                        std_avg[...,None],std_diff[...,None]),dim=-1)
    if features.shape!=(n,n,10) or not torch.isfinite(features).all():
        raise FloatingPointError("pair feature construction failed")
    return 0.5*(features+features.transpose(0,1))


class PairRelationMLP(nn.Module):
    """Shared 10→32→1 pair correction, initialized to exact zero residual."""
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(10,32)
        self.output=nn.Linear(32,1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, e0, weight_a):
        if features.ndim!=3 or features.shape[-1]!=10 or e0.shape!=features.shape[:2] or weight_a.shape!=e0.shape:
            raise ValueError("MLP input shapes must be [N,N,10], [N,N], [N,N]")
        r=self.output(torch.relu(self.hidden(features))).squeeze(-1)
        r=(r+r.T)*0.5
        correction=weight_a*r
        correction=correction-torch.diag_embed(torch.diagonal(correction))
        return e0+correction, r


def energy_components(q, source_a, gain):
    """Return E0 and wA using the registered operator's exact arithmetic."""
    q_cpu=q.detach().cpu().numpy() if torch.is_tensor(q) else np.asarray(q)
    a_cpu=source_a.detach().cpu().numpy() if torch.is_tensor(source_a) else np.asarray(source_a)
    target_device=q.device if torch.is_tensor(q) else torch.device("cpu")
    e0=torch.as_tensor(base_energy(q_cpu,a_cpu,gain),dtype=torch.float64,device=target_device)
    a=torch.as_tensor(a_cpu,device=target_device,dtype=torch.float64)
    w=1.0+1024.0*(a+a.T)/(2.0*float(gain))
    w=(w+w.T)*0.5
    w=w-torch.diag_embed(torch.diagonal(w))
    return e0,w


def feasible_target(source_labels, gt32, *, grid_shape=None):
    """Split each GT id (including zero) into 4-connected components inside F."""
    support=np.asarray(source_labels)
    gt=np.asarray(gt32)
    if support.ndim!=2 or gt.shape!=support.shape or not np.issubdtype(support.dtype,np.integer) or not np.issubdtype(gt.dtype,np.integer):
        raise ValueError("source labels and modal GT must be same-shape integer grids")
    h,w=support.shape
    if grid_shape is not None and _grid_shape(grid_shape)!=(h,w): raise ValueError("grid shape mismatch")
    if support.size==0 or np.any(support<0) or np.any(gt<0): raise ValueError("labels must be nonempty/nonnegative")
    f=(support>0); target=np.zeros((h,w),np.int32); component_id=0
    remaining=f.copy()
    for gt_id in np.unique(gt[f]):
        mask=f & (gt==gt_id)
        for seed in np.flatnonzero(mask):
            r,c=divmod(int(seed),w)
            if not mask[r,c] or not remaining[r,c]: continue
            component_id+=1; remaining[r,c]=False; todo=[(r,c)]
            target[r,c]=component_id
            while todo:
                rr,cc=todo.pop()
                for nr,nc in ((rr-1,cc),(rr+1,cc),(rr,cc-1),(rr,cc+1)):
                    if 0<=nr<h and 0<=nc<w and mask[nr,nc] and remaining[nr,nc]:
                        remaining[nr,nc]=False; target[nr,nc]=component_id; todo.append((nr,nc))
    if not np.array_equal(target>0,f): raise AssertionError("feasible targets must preserve every fixed-support node")
    return target


def pair_targets(target, support):
    target_tensor=torch.as_tensor(target)
    t=target_tensor.reshape(-1)
    support_tensor=torch.as_tensor(support,device=t.device,dtype=torch.bool)
    if tuple(target_tensor.shape)!=tuple(support_tensor.shape):
        raise ValueError("target and fixed support must have identical geometry")
    f=support_tensor.reshape(-1)
    if not f.any(): return torch.zeros((0,0),dtype=torch.float32,device=t.device),torch.zeros((0,0),dtype=torch.float32,device=t.device),torch.empty(0,dtype=torch.long,device=t.device)
    idx=torch.nonzero(f,as_tuple=False).flatten(); labels=t[idx]
    same=(labels[:,None]==labels[None,:]).to(torch.float32)
    upper=torch.triu(torch.ones_like(same,dtype=torch.bool),diagonal=1)
    pos=upper & (same>0); neg=upper & (same==0)
    pos_count=int(pos.sum()); neg_count=int(neg.sum())
    weights=torch.zeros_like(same)
    if pos_count and neg_count:
        weights[pos]=0.5/pos_count; weights[neg]=0.5/neg_count
    elif pos_count: weights[pos]=1.0/pos_count
    elif neg_count: weights[neg]=1.0/neg_count
    return same,weights,idx


def structured_hinge(e_theta,e0,target,support):
    """Approximate structured hinge with detached CPU oracle memberships."""
    same,weights,idx=pair_targets(target,support)
    if idx.numel()<2 or not weights.any(): return e_theta.sum()*0.0
    et=e_theta.index_select(0,idx).index_select(1,idx)
    ebase=e0.index_select(0,idx).index_select(1,idx)
    upper=torch.triu(torch.ones_like(ebase,dtype=torch.bool),diagonal=1)
    z=ebase.detach()[upper].abs().sum().clamp_min(1.0)
    y=same.to(et.device); a=weights.to(et.device)
    # Preserve original grid coordinates: compacting F would create false
    # four-neighbour edges across background gaps.
    support_np=np.asarray(support.detach().cpu() if torch.is_tensor(support) else support,dtype=bool)
    n=e_theta.shape[0]; full=torch.zeros((n,n),device=et.device,dtype=et.dtype)
    aug_upper=et.detach()[upper]+z*a[upper]*(1.0-2.0*y[upper])
    compact_aug=torch.zeros_like(et.detach())
    compact_aug[upper]=aug_upper
    compact_aug=compact_aug+compact_aug.T
    full[idx[:,None],idx[None,:]]=compact_aug
    pred=solve_energy(full.cpu().numpy(),support_np.astype(np.int32),grid_shape=support_np.shape)["labels"].reshape(-1)
    pred_f=pred[idx.cpu().numpy()]
    yhat=torch.as_tensor((pred_f[:,None]==pred_f[None,:]),device=et.device,dtype=et.dtype)
    score=(et[upper]*yhat[upper]).sum()/z
    dist=(a[upper]*(yhat[upper]-y[upper]).abs()).sum()
    gold=(et[upper]*y[upper]).sum()/z
    return torch.maximum(score+dist,gold)-gold


def local_margin_loss(e_theta,e0,target,support):
    same,weights,idx=pair_targets(target,support)
    if idx.numel()<2 or not weights.any(): return e_theta.sum()*0.0
    et=e_theta.index_select(0,idx).index_select(1,idx)
    ebase=e0.index_select(0,idx).index_select(1,idx)
    upper=torch.triu(torch.ones_like(ebase,dtype=torch.bool),diagonal=1)
    z=ebase.detach()[upper].abs().mean().clamp_min(1.0)
    y=same.to(et.device); a=weights.to(et.device)
    return (a[upper]*torch.relu(1.0-(2*y[upper]-1.0)*et[upper]/z)).sum()
