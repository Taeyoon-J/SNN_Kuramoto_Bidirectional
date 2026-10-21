"""Parameter-replication warm start, grid 32 -> grid 64.

Ported from the peer's SW_0154, which states the rule exactly: "A row-major native16
node maps to its exact 4x4 block of native64 children. Node-leading parameters repeat
by that mapping; pairwise parameters use it on both axes."

Here the step is 32 -> 64, so each parent node (r, c) owns the 2x2 block
{(2r, 2c), (2r, 2c+1), (2r+1, 2c), (2r+1, 2c+1)}.

Why this matters: PV2_0120 trained grid 64 FROM SCRATCH and did not finish one epoch in
8.5 hours, projecting to about 14 days. The peer reports native64 at roughly 2 hours a
seed (SW_0192) because they never train it from scratch -- they replicate a trained
model's parameters onto the finer grid and fine-tune briefly. REVIEW_0006 recorded this
technique as worth taking and then I did not use it when route C was designed.

grid_distance is NOT replicated. It is a geometric quantity the model rebuilds from its
own grid, and replicating a coarse one would hand the finer model the coarse geometry.
"""
import argparse, torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--src-grid", type=int, default=32)
ap.add_argument("--dst-grid", type=int, default=64)
args = ap.parse_args()

GS, GD = args.src_grid, args.dst_grid
assert GD % GS == 0, "destination grid must be a multiple of the source"
F = GD // GS
NS, ND = GS * GS, GD * GD

# parent[i64] = the grid-32 node whose block contains grid-64 node i64
rows = torch.arange(GD).view(-1, 1).expand(GD, GD) // F
cols = torch.arange(GD).view(1, -1).expand(GD, GD) // F
parent = (rows * GS + cols).reshape(-1)                 # [ND] in [0, NS)
assert parent.min() == 0 and parent.max() == NS - 1

sd = torch.load(args.src, map_location="cpu")
sd = sd.get("state_dict", sd)
out, report = {}, []
for k, v in sd.items():
    if not hasattr(v, "shape"):
        out[k] = v; continue
    s = tuple(v.shape)
    if k.endswith("grid_distance"):
        # Geometric, not learned: replicating the coarse grid's distances would hand
        # the finer model the coarse geometry. But a strict load_state_dict needs the
        # key present, so it is filled with the DESTINATION grid's own distances,
        # computed here the way the model computes them.
        rr = torch.arange(GD).view(-1, 1).expand(GD, GD).reshape(-1).float()
        cc = torch.arange(GD).view(1, -1).expand(GD, GD).reshape(-1).float()
        d = ((rr[:, None] - rr[None, :]) ** 2 + (cc[:, None] - cc[None, :]) ** 2).sqrt()
        out[k] = d.to(v.dtype)
        report.append((k, s, "REBUILT for grid %d -> %s" % (GD, tuple(d.shape))))
        continue
    if s.count(NS) >= 2:
        w = v.index_select(0, parent).index_select(1, parent)
        out[k] = w; report.append((k, s, "pairwise  -> %s" % (tuple(w.shape),)))
    elif NS in s:
        ax = s.index(NS)
        w = v.index_select(ax, parent)
        out[k] = w; report.append((k, s, "node axis %d -> %s" % (ax, tuple(w.shape))))
    else:
        out[k] = v; report.append((k, s, "copied unchanged"))

print("grid %d -> %d  (each parent owns a %dx%d block)" % (GS, GD, F, F))
for k, s, how in report:
    print("  %-44s %-14s %s" % (k, str(s), how))
torch.save(out, args.dst)
print("wrote %s" % args.dst)
