"""Pure CPU conversion from the registered source affinity Q to native32 QCC."""
from __future__ import annotations

import numpy as np
import torch

from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels


def qcc_labels_from_affinity(q, *, grid_shape=(32, 32), threshold=0.50, min_group_size=8):
    """Apply production full-graph QCC threshold/component/background rules.

    This consumes the already-computed source affinity Q; it does not infer Q
    from aggregate activity, inspect masks, or choose a scene/object count.
    Components are discovered in increasing node order, filtered by the
    production minimum size, stably sorted by descending size, and the first
    largest is omitted as background. The registered native32 label converter
    assigns the remaining group IDs.
    """
    if (not isinstance(grid_shape, (tuple, list)) or len(grid_shape) != 2
            or any(type(x) is not int or x <= 0 for x in grid_shape)):
        raise ValueError("grid_shape must contain two positive integers")
    if not np.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1:
        raise ValueError("threshold must be finite and in [0,1]")
    if type(min_group_size) is not int or min_group_size < 1:
        raise ValueError("min_group_size must be a positive integer")
    h,w=grid_shape; n=h*w
    if torch.is_tensor(q):
        if q.dtype!=torch.float32:
            raise ValueError("Q must preserve the registered float32 source-cache dtype")
        affinity=q.detach().to(device="cpu")
    else:
        source=np.asarray(q)
        if source.dtype!=np.float32:
            raise ValueError("Q must preserve the registered float32 source-cache dtype")
        affinity=torch.from_numpy(np.ascontiguousarray(source))
    if affinity.ndim!=3 or tuple(affinity.shape[1:])!=(n,n):
        raise ValueError("Q must have shape [B,N,N] matching grid_shape")
    if affinity.shape[0]<1 or not torch.isfinite(affinity).all():
        raise ValueError("Q batch must be nonempty and finite")
    if torch.any(affinity < -1e-7) or torch.any(affinity > 1+1e-7):
        raise ValueError("Q entries must lie in [0,1]")
    if not torch.allclose(affinity,affinity.transpose(1,2),rtol=0,atol=2e-6):
        raise ValueError("Q must be symmetric")

    groups_by_batch=[]
    for b in range(affinity.shape[0]):
        linked=affinity[b]>=float(threshold)
        seen=np.zeros(n,dtype=bool)
        groups=[]
        for start in range(n):
            if seen[start]:
                continue
            seen[start]=True
            frontier=[start]
            members=[start]
            while frontier:
                node=frontier.pop()
                # torch nonzero matches production's ascending node order.
                neighbors=torch.nonzero(linked[node] & ~torch.from_numpy(seen),as_tuple=False).flatten().tolist()
                for other in neighbors:
                    seen[other]=True
                    members.append(other)
                    frontier.append(other)
            if len(members)>=min_group_size:
                groups.append(tuple(sorted(members)))
        # Python's stable sort preserves increasing-start discovery on ties,
        # matching spike_synchrony_components exactly.
        groups.sort(key=len,reverse=True)
        if groups:
            groups=groups[1:]
        groups_by_batch.append(groups)
    return spatial_components_to_patch_labels(groups_by_batch,(h,w),device="cpu")
