"""Which costs more foreground IoU: the patches we miss, or the ones we invent?

Measured directly from the saved predictions and the targets -- no modelling.
ORACLE DIAGNOSTIC.
"""
import json
import numpy as np, torch
man = json.load(open("/export_home/tkim1/collaborative_test/data/split_manifest.json"))
vs, _ = man["ranges"]["validation"]
lab = torch.load("/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt",
                 map_location="cpu")["patch_labels"]
pred = torch.as_tensor(np.load("/work/USERS/tkim1/figures/cov_diag_pred.npz",
                               allow_pickle=True)["patch_labels"]).long()
LIM = pred.size(0)
TP = FP = FN = TN = 0
for r in range(LIM):
    t = (lab[vs + r].reshape(-1) > 0).numpy()
    p = (pred[r].reshape(-1) > 0).numpy()
    TP += int((t & p).sum()); FP += int((~t & p).sum())
    FN += int((t & ~p).sum()); TN += int((~t & ~p).sum())
n = LIM
print("per image, grid 32 (1024 patches)")
print("  true foreground        %6.1f" % (( TP + FN)/n))
print("  predicted foreground   %6.1f" % (( TP + FP)/n))
print("  true positives         %6.1f" % (TP/n))
print("  FALSE POSITIVES        %6.1f   (%.1f%% of what we predict)" % (FP/n, 100*FP/(TP+FP)))
print("  false negatives        %6.1f   (%.1f%% of what exists)"     % (FN/n, 100*FN/(TP+FN)))
print()
iou = lambda i, u: i/u
cur = TP/(TP+FP+FN)
print("foreground IoU now                              %.4f" % cur)
print("  if every MISSED patch were recovered          %.4f   (+%.4f)"
      % ((TP+FN)/((TP+FN)+FP), (TP+FN)/((TP+FN)+FP) - cur))
print("  if every FALSE POSITIVE were removed          %.4f   (+%.4f)"
      % (TP/(TP+FN), TP/(TP+FN) - cur))
print("  if both                                       1.0000")
