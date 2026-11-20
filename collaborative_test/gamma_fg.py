"""Does the gamma drive carry figure/ground, label-free?

Every PLV loss in this project shapes PARTITION structure -- bimodality, balance,
collapse, participation ratio. None of them says which block is foreground, which
is why PV2_0114 measured internal coherence at AUC 0.500 for real against
spurious groups: the optimiser made the background block coherent too.

A figure/ground term needs a label-free cue. gamma is the input drive and is
available during training. This measures whether gamma, and its local deviation
from its spatial neighbourhood, separates true foreground patches from true
background ones. If it does not, no gamma-based term is worth writing.

ORACLE DIAGNOSTIC -- the truth scores the cue, it never produces a mask.
No forward pass.
"""
import json
import numpy as np, torch
import torch.nn.functional as F

GRID = 32
man = json.load(open("/export_home/tkim1/collaborative_test/data/split_manifest.json"))
vs, _ = man["ranges"]["validation"]
gamma = torch.load("/work/USERS/tkim1/gamma_sequences/wm10k_full_grid32.pt", map_location="cpu").float()
lab = torch.load("/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt", map_location="cpu")["patch_labels"]
LIM = 200
print("gamma", tuple(gamma.shape))

def auc(score, pos):
    r = np.argsort(np.argsort(score))
    npos = pos.sum(); nneg = len(pos) - npos
    if npos == 0 or nneg == 0: return float("nan")
    return (r[pos].sum() - npos*(npos-1)/2) / (npos*nneg)

cues = {k: [] for k in ("gamma_norm", "local_dev", "local_dev_k3", "grad_mag")}
ys = []
for r in range(LIM):
    g = gamma[vs + r]                      # [C, N] or [N, C]
    g = g if g.shape[0] <= 16 else g.t()   # -> [C, N]
    C = g.shape[0]
    m = g.reshape(C, 1, GRID, GRID)
    blur5 = F.avg_pool2d(F.pad(m, (2,2,2,2), mode="replicate"), 5, stride=1)
    blur3 = F.avg_pool2d(F.pad(m, (1,1,1,1), mode="replicate"), 3, stride=1)
    gx = m[..., :, 1:] - m[..., :, :-1]
    gy = m[..., 1:, :] - m[..., :-1, :]
    gx = F.pad(gx, (0,1,0,0)); gy = F.pad(gy, (0,0,0,1))
    cues["gamma_norm"].append(m.reshape(C, -1).norm(dim=0).numpy())
    cues["local_dev"].append((m - blur5).reshape(C, -1).norm(dim=0).numpy())
    cues["local_dev_k3"].append((m - blur3).reshape(C, -1).norm(dim=0).numpy())
    cues["grad_mag"].append((gx.pow(2) + gy.pow(2)).sqrt().reshape(C, -1).sum(0).numpy())
    ys.append((lab[vs + r].reshape(-1) > 0).numpy())

y = np.concatenate(ys)
print("patches %d   true foreground %.1f%%" % (len(y), 100*y.mean()))
print()
print("cue                         AUC vs true foreground")
best = None
for k, v in cues.items():
    s = np.concatenate(v); a = auc(s, y)
    print("  %-22s      %.3f" % (k, a))
    if best is None or abs(a-0.5) > abs(best[1]-0.5): best = (k, a)
print()
k, a = best
s = np.concatenate(cues[k])
print("best cue: %s (AUC %.3f).  precision when taking the top q by this cue:" % (k, a))
order = np.argsort(-s if a > 0.5 else s)
for q in (0.08, 0.12, 0.16, 0.25):
    n = int(q*len(s))
    print("   top %4.0f%%  ->  %5.1f%% true foreground   (base %.1f%%),  recall %.1f%%"
          % (100*q, 100*y[order[:n]].mean(), 100*y.mean(),
             100*y[order[:n]].sum()/y.sum()))
