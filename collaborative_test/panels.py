"""Render the pipeline's actual intermediate representations as figure panels.

Not illustrations: every panel is the real tensor from one forward pass of the published
checkpoint on one validation image, so the figure shows what the model does rather than
what it is supposed to do.

The image is chosen as the one whose fg_ari is closest to the split mean, so the
qualitative panel is representative rather than picked for looking good; its own score is
printed and goes in the caption.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.colors as mcolors
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

sys.path.insert(0, "/export_home/tkim1/collaborative_test")
sys.path.insert(0, "/export_home/tkim1")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")

OUT = Path("/work/USERS/tkim1/figpanels"); OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda"
GRID = 16
IMGDIR = Path("/work/USERS/tkim1/clevr/with_masks/images10k")

from hyperparameter import S2NetHyperparameters
from s2net_cls import S2NetCore
from evaluation import evaluate_patch_masks

man = json.load(open("/export_home/tkim1/collaborative_test/data/split_manifest.json"))
vs, ve = man["ranges"]["validation"]
gamma_all = torch.load("/work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt", map_location="cpu").float()
tgt_all = torch.load("/work/USERS/tkim1/clevr/with_masks/targets_v1.pt", map_location="cpu")["patch_labels"]

# pick a representative image from the saved predictions
pred = torch.as_tensor(np.load("/work/USERS/tkim1/figures/pred_grid16_cc.npz",
                               allow_pickle=True)["patch_labels"]).long()
tv = tgt_all[vs:vs + pred.size(0)]
per = evaluate_patch_masks(pred, tv)["per_image"]["fg_ari"].numpy()
ok = ~np.isnan(per)
mean = per[ok].mean()
idx = int(np.nanargmin(np.abs(np.where(ok, per, np.inf) - mean)))
gid = vs + idx
print("representative image: validation row %d (id %d), fg_ari %.4f vs split mean %.4f"
      % (idx, gid, per[idx], mean))

hp = S2NetHyperparameters(
    num_feature_maps=8, num_regions=256, sc=None, gamma_drive_mode="static",
    num_time_steps=1024, theta_init="gamma", gamma_phase_mode="standardize_tanh",
    osc_dim=4, freq_gain=2.0, graph_mode="learned", graph_top_k=32, k=256.0,
    graph_spatial_decay=0.35, low_n=-4., high_n=0., membrane_vth=0.06,
    membrane_low_m=-4., membrane_high_m=0., gate_mode="raw", spike_per_component=True,
    geodesic_steps=3, geodesic_radius=1.5, geodesic_contrast=2.0,
    spike_classify_method="spatial_components", spike_spatial_grid_size=(GRID, GRID))
core = S2NetCore(hp, device=DEV).to(DEV)
core.load_state_dict(torch.load("/work/USERS/tkim1/runs/BIM6_s1/core.pt", map_location=DEV))
core.gamma_channels = 8
core.gamma_graph_extra = 0
core.eval()

g = gamma_all[gid:gid + 1].to(DEV)
with torch.no_grad():
    A = core.graph_generator(g.transpose(1, 2))
    _, spikes, membrane, theta = core(g, return_core_out=True, return_theta=True)
print("A", tuple(A.shape), "spikes", tuple(spikes.shape),
      "theta", None if theta is None else tuple(theta.shape))

def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", pad_inches=0.02,
                transparent=True)
    plt.close(fig)
    print("  wrote", name)

# 1. input
im = Image.open(IMGDIR / ("CLEVR_masks_%06d.png" % gid)).convert("RGB").resize((128, 128), Image.BILINEAR)
fig, ax = plt.subplots(figsize=(2.2, 2.2)); ax.imshow(np.asarray(im)); ax.axis("off")
save(fig, "p1_input.png")

# 2. gamma, 8 channels on the 16x16 grid
gm = g[0].cpu().numpy().reshape(8, GRID, GRID)
fig, axes = plt.subplots(2, 4, figsize=(2.2, 1.15))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(gm[i], cmap="gray", interpolation="nearest"); ax.axis("off")
fig.subplots_adjust(wspace=0.08, hspace=0.08)
save(fig, "p2_gamma.png")

# 3. coupling graph A, shown as a 64x64 corner zoom. The full 256x256 matrix is black
# on a linear scale because its mass sits in a few near-diagonal bands -- the grid's own
# +-1 and +-16 neighbours. The zoom makes one band legible; log norm from the 75th
# percentile up is a display choice only.
Anp = A[0].detach().cpu().numpy()[:64, :64]
fig, ax = plt.subplots(figsize=(2.2, 2.2))
ax.imshow(Anp, cmap="magma", interpolation="nearest",
          norm=mcolors.LogNorm(vmin=max(np.percentile(Anp, 75), 1e-12), vmax=Anp.max()))
ax.axis("off")
save(fig, "p3_graph.png")

# 4. final phase
if theta is not None:
    th = theta[-1] if theta.dim() == 4 else theta
    th = th[0, :, 0].cpu().numpy().reshape(GRID, GRID)
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.imshow(np.mod(th, 2 * np.pi), cmap="twilight", interpolation="nearest"); ax.axis("off")
    save(fig, "p4_phase.png")

# 5. spike raster. The model's spike variable is a GRADED value in [0,1] -- the
# surrogate the gradient needs -- not a binary train: only 1.1% of its entries are
# exactly 0 or 1. Drawn as it is, with the 64 most active patches over a 192-step
# window so individual events are a cell rather than a sub-pixel smear.
sp = spikes[0].cpu().numpy()                      # [patches, time]
order = np.argsort(-sp.mean(1))[:64]
win = sp[np.sort(order)][:, -192:]
fig, ax = plt.subplots(figsize=(2.2, 2.2))
ax.imshow(win, cmap="binary", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
ax.axis("off")
save(fig, "p5_spikes.png")

# 6/7. prediction and ground truth
rng = np.random.default_rng(0)
cols = rng.uniform(0.3, 0.95, size=(24, 3)); cols[0] = (0.10, 0.10, 0.12)
cmap = ListedColormap(cols)
def relabel(a):
    o = np.zeros_like(a)
    for n, v in enumerate([x for x in np.unique(a) if x != 0], start=1): o[a == v] = n
    return o
for name, arr in (("p6_pred.png", pred[idx].numpy()), ("p7_target.png", tv[idx].numpy())):
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.imshow(relabel(arr), cmap=cmap, vmin=0, vmax=23, interpolation="nearest"); ax.axis("off")
    save(fig, name)

json.dump({"id": int(gid), "row": idx, "fg_ari": float(per[idx]), "split_mean": float(mean)},
          open(OUT / "meta.json", "w"), indent=2)
print("done")
