"""PV2_0115's mechanism check, registered before the run and cheaper than the score.

The claim the term makes is narrow and measurable: figure/ground sits in the input
at AUC 0.947 (a patch's gamma deviation from its 5x5 neighbourhood, over 200 images)
and reaches the spikes at 0.494, which is chance. spike_gamma_contrast_loss asks for
it back. So before any patch metric is read, this asks whether mean spike activity
now separates true foreground from background better than 0.494.

If the AUC has not moved, the term did not do what it is documented to do and the
three metrics are not interpreted -- that is the gate as it was written into
pv2_0115.sh before the arms trained.

Reads the truth to SCORE A SIGNAL, never to produce a mask. Oracle diagnostic.
"""
import json, sys, argparse
import numpy as np, torch
sys.path.insert(0, "/export_home/tkim1/collaborative_test")
sys.path.insert(0, "/export_home/tkim1")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")
from hyperparameter import S2NetHyperparameters
from s2net_cls import S2NetCore

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoints", nargs="+",
                default=["GCON_ctrl", "GCON_w1", "GCON_w5"])
ap.add_argument("--limit", type=int, default=60)
ap.add_argument("--grid", type=int, default=16)
ap.add_argument("--gamma", default="/work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt")
ap.add_argument("--targets", default="/work/USERS/tkim1/clevr/with_masks/targets_v1.pt")
args = ap.parse_args()

GRID = args.grid
N = GRID * GRID
DEV = "cuda" if torch.cuda.is_available() else "cpu"
man = json.load(open("/export_home/tkim1/collaborative_test/data/split_manifest.json"))
vs, _ = man["ranges"]["validation"]
gamma = torch.load(args.gamma, map_location="cpu").float()
lab = torch.load(args.targets, map_location="cpu")["patch_labels"]

def auc(score, positive):
    rank = np.argsort(np.argsort(score))
    npos = int(positive.sum()); nneg = len(positive) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return (rank[positive].sum() - npos * (npos - 1) / 2) / (npos * nneg)

def build(name):
    hp = S2NetHyperparameters(
        num_feature_maps=8, num_regions=N, sc=None, gamma_drive_mode="static",
        num_time_steps=1024, theta_init="gamma", gamma_phase_mode="standardize_tanh",
        osc_dim=4, freq_gain=2.0, graph_mode="learned", graph_top_k=32, k=float(N),
        graph_spatial_decay=0.35, low_n=-4., high_n=0., membrane_vth=0.06,
        membrane_low_m=-4., membrane_high_m=0., gate_mode="raw",
        spike_per_component=True, geodesic_steps=3, geodesic_radius=1.5,
        geodesic_contrast=2.0, geodesic_chunk=64,
        spike_classify_method="spatial_components",
        spike_spatial_grid_size=(GRID, GRID)).validate()
    core = S2NetCore(hp, device=DEV).to(DEV)
    core.load_state_dict(torch.load(
        "/work/USERS/tkim1/runs/%s/core.pt" % name, map_location=DEV))
    core.gamma_channels = 8
    core.gamma_graph_extra = 0
    core.eval()
    return core

print("activity-vs-foreground AUC, %d validation images, grid %d" % (args.limit, GRID))
print("  reference measured before the run:  gamma deviation 0.947,  spike activity 0.494")
print()
print("  %-14s  AUC     mean act FG   mean act BG   separation" % "checkpoint")
for name in args.checkpoints:
    try:
        core = build(name)
    except Exception as exc:
        print("  %-14s  FAILED: %s: %s" % (name, type(exc).__name__, exc))
        continue
    acts, ys = [], []
    for r in range(args.limit):
        g = gamma[vs + r:vs + r + 1].to(DEV)
        with torch.no_grad():
            _, spikes, _, _ = core(g, return_core_out=True, return_theta=True)
        acts.append(spikes[0, :, 512:].float().mean(-1).cpu().numpy())
        ys.append((lab[vs + r].reshape(-1) > 0).numpy())
    a = np.concatenate(acts)
    y = np.concatenate(ys).astype(bool)
    print("  %-14s  %.3f   %.4f        %.4f        %+.4f"
          % (name, auc(a, y), a[y].mean(), a[~y].mean(), a[y].mean() - a[~y].mean()))
print()
print("  GATE: the treatment arms must beat the control's AUC. If they do not, the")
print("        term did not do what it is documented to do and no patch metric from")
print("        PV2_0115 is interpreted.")
