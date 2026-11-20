"""Row x column panels for the module figure: how each oscillator group locks in over time.

The analogue of Slot Attention's "slots compete for keys" columns. A row is one of the
final groups the readout returns; a column is a time step of the same rollout. The cell
is cos(theta_i(t) - mean theta of that group at t) -- how aligned every oscillator on the
grid is with that group's phase, right then. Nothing here is drawn by hand and nothing
uses the ground truth: the groups come from the model's own spikes.

theta comes back as [B, T, N, D], not [T, B, N, D]; indexing it the other way silently
gives T = 1 and four identical columns.
"""
import json, sys
from pathlib import Path
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/export_home/tkim1/collaborative_test")
sys.path.insert(0, "/export_home/tkim1")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")
OUT = Path("/work/USERS/tkim1/figpanels"); OUT.mkdir(parents=True, exist_ok=True)
DEV, GRID = "cuda", 16

from hyperparameter import S2NetHyperparameters
from s2net_cls import S2NetCore

man = json.load(open("/export_home/tkim1/collaborative_test/data/split_manifest.json"))
vs, ve = man["ranges"]["validation"]
gamma_all = torch.load("/work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt", map_location="cpu").float()
meta = json.load(open(OUT / "meta.json"))
idx, gid = meta["row"], meta["id"]
pred = torch.as_tensor(np.load("/work/USERS/tkim1/figures/pred_grid16_cc.npz",
                               allow_pickle=True)["patch_labels"]).long()[idx].reshape(-1)

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
core.gamma_channels = 8; core.gamma_graph_extra = 0; core.eval()

g = gamma_all[gid:gid + 1].to(DEV)
with torch.no_grad():
    _, spikes, membrane, theta = core(g, return_core_out=True, return_theta=True)
print("theta", tuple(theta.shape), "spikes", tuple(spikes.shape))

th = theta if theta.dim() == 4 else theta.unsqueeze(0)   # [B, T, N, D]
T = th.shape[1]
bar = th[0].mean(dim=-1)                                 # [T, N]  phase per node
print("T =", T, "bar", tuple(bar.shape))

labels = [int(v) for v in torch.unique(pred) if int(v) != 0]
sizes = sorted(((int((pred == v).sum()), v) for v in labels), reverse=True)
top = [v for _, v in sizes[:4]]
print("groups", sizes[:6], "-> rows", top)

steps = [0, T // 8, T // 2, T - 1]
json.dump({"steps": [int(s) + 1 for s in steps], "T": int(T),
           "group_sizes": [[int(n), int(v)] for n, v in sizes[:4]]},
          open(OUT / "rowmeta.json", "w"), indent=2)

for r, v in enumerate(top):
    nodes = torch.where(pred == v)[0].to(DEV)
    for c, t in enumerate(steps):
        p = bar[t]                                        # [N]
        m = torch.atan2(torch.sin(p[nodes]).mean(), torch.cos(p[nodes]).mean())
        align = torch.cos(p - m).cpu().numpy().reshape(GRID, GRID)
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        ax.imshow(align, cmap="viridis", vmin=-1, vmax=1, interpolation="nearest")
        ax.axis("off")
        fig.savefig(OUT / ("row%d_t%d.png" % (r + 1, c + 1)), dpi=150,
                    bbox_inches="tight", pad_inches=0.0, transparent=True)
        plt.close(fig)
    print("  row", r + 1, "label", v, "n", int((pred == v).sum()))
print("done")
