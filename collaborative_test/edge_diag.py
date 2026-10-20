"""Are the missed object patches the ambiguous boundary ones?

PV2_0113 found the coverage deficit is 5-8% of nearly every object -- its
boundary -- not whole small objects. At grid 32 a boundary cell is part object
and part background, and its target label is decided by MAJORITY VOTE. The cell's
`patch_purity`, already stored in the targets, is the fraction of its pixels that
belong to the label it was given.

If the patches the readout misses are overwhelmingly low-purity, then part of the
remaining gap is the patch grid's own ambiguity rather than a model failure, and
PV2_0106's oracle ceiling is not reachable by a better model at this grid.

Reads the truth: ORACLE DIAGNOSTIC, never a score. No forward pass.
"""
import json
import numpy as np, torch

man = json.load(open("/export_home/tkim1/collaborative_test/data/split_manifest.json"))
vs, _ = man["ranges"]["validation"]
d = torch.load("/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt", map_location="cpu")
lab, pur, ties = d["patch_labels"], d["patch_purity"], d["patch_ties"]
pred = torch.as_tensor(np.load("/work/USERS/tkim1/figures/cov_diag_pred.npz",
                               allow_pickle=True)["patch_labels"]).long()
LIM = pred.size(0)

cov_p, mis_p, bg_p = [], [], []
cov_t = mis_t = 0
for r in range(LIM):
    t = lab[vs + r].reshape(-1); p = pur[vs + r].reshape(-1).numpy()
    claimed = (pred[r].reshape(-1) > 0).numpy()
    fg = (t > 0).numpy()
    cov_p.append(p[fg & claimed]); mis_p.append(p[fg & ~claimed])
    bg_p.append(p[~fg])
    cov_t += int((fg & claimed).sum()); mis_t += int((fg & ~claimed).sum())

c, m, b = np.concatenate(cov_p), np.concatenate(mis_p), np.concatenate(bg_p)
print("images %d   foreground patches %d   covered %d   MISSED %d (%.1f%%)"
      % (LIM, cov_t + mis_t, cov_t, mis_t, 100*mis_t/(cov_t+mis_t)))
print()
print("patch_purity = fraction of the cell's pixels belonging to its assigned label")
print("                      n      mean     p10      p25      p50      p75")
for nm, a in (("covered FG ", c), ("MISSED  FG ", m), ("background ", b)):
    print("  %s %8d   %.4f   %.4f   %.4f   %.4f   %.4f"
          % (nm, len(a), a.mean(), *np.percentile(a, [10, 25, 50, 75])))
print()
print("fraction of each population that is a NARROW-majority cell (purity < 0.6):")
for nm, a in (("covered FG", c), ("MISSED  FG", m)):
    print("  %s  %.1f%%" % (nm, 100*(a < 0.6).mean()))
print()
print("if every missed patch with purity < X is counted as grid ambiguity, the")
print("share of the miss that is NOT a model failure:")
for x in (0.55, 0.60, 0.70):
    print("   purity < %.2f   ->  %.1f%% of the missed patches" % (x, 100*(m < x).mean()))
print()
print("ties (cells where two labels tied) among missed: %.1f%%  among covered: %.1f%%"
      % (100*np.concatenate([ties[vs+r].reshape(-1).numpy()[(lab[vs+r].reshape(-1)>0).numpy() & ~(pred[r].reshape(-1)>0).numpy()] for r in range(LIM)]).mean(),
         100*np.concatenate([ties[vs+r].reshape(-1).numpy()[(lab[vs+r].reshape(-1)>0).numpy() & (pred[r].reshape(-1)>0).numpy()] for r in range(LIM)]).mean()))
