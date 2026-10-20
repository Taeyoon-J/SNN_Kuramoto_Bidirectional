"""Does spike activity tell foreground from background at all?

Reads the PRODUCTION readout's saved predictions for the claimed set, and a
forward pass for the activity. Uses the ground truth: ORACLE DIAGNOSTIC, never a
score.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/export_home/tkim1/collaborative_test")
sys.path.insert(0, "/export_home/tkim1")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")
from hyperparameter import S2NetHyperparameters
from s2net_cls import S2NetCore

DEV, GRID, N, LIM = "cuda", 32, 1024, 60
man = json.load(open("/export_home/tkim1/collaborative_test/data/split_manifest.json"))
vs, _ = man["ranges"]["validation"]
gamma = torch.load("/work/USERS/tkim1/gamma_sequences/wm10k_full_grid32.pt", map_location="cpu").float()
tgt = torch.load("/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt", map_location="cpu")["patch_labels"]
pred = torch.as_tensor(np.load("/work/USERS/tkim1/figures/cov_diag_pred.npz",
                               allow_pickle=True)["patch_labels"]).long()

hp = S2NetHyperparameters(
    num_feature_maps=8, num_regions=N, sc=None, gamma_drive_mode="static",
    num_time_steps=1024, theta_init="gamma", gamma_phase_mode="standardize_tanh",
    osc_dim=4, freq_gain=2.0, graph_mode="learned", graph_top_k=32, k=float(N),
    graph_spatial_decay=0.35, low_n=-4., high_n=0., membrane_vth=0.06,
    membrane_low_m=-4., membrane_high_m=0., gate_mode="raw", spike_per_component=True,
    membrane_threshold_mode="population", membrane_threshold_k=0.5,
    geodesic_steps=3, geodesic_radius=1.5, geodesic_contrast=2.0, geodesic_chunk=16,
    spike_classify_method="spatial_components", spike_spatial_grid_size=(GRID, GRID)).validate()
core = S2NetCore(hp, device=DEV).to(DEV)
core.load_state_dict(torch.load("/work/USERS/tkim1/runs/GC_w0/core.pt", map_location=DEV))
core.gamma_channels = 8; core.gamma_graph_extra = 0; core.eval()

uf, ub, tot, cl = [], [], 0, 0
for r in range(min(LIM, pred.size(0))):
    g = gamma[vs + r:vs + r + 1].to(DEV)
    with torch.no_grad():
        _, spikes, _, _ = core(g, return_core_out=True, return_theta=True)
    act = spikes[0, :, 512:].float().mean(-1).cpu()
    claimed = pred[r].reshape(-1) > 0
    truth = tgt[vs + r].reshape(-1) > 0
    tot += int(truth.sum()); cl += int((claimed & truth).sum())
    uf.append(act[(~claimed) & truth].numpy())
    ub.append(act[(~claimed) & (~truth)].numpy())

f, b = np.concatenate(uf), np.concatenate(ub)
print("images %d  true fg %d  claimed by readout %d (%.1f%%)  MISSED %d (%.1f%%)"
      % (LIM, tot, cl, 100*cl/tot, tot-cl, 100*(tot-cl)/tot))
print()
print("activity of UNCLAIMED patches        n      mean     p50      p90")
for nm, a in (("true FG", f), ("true BG", b)):
    print("  %s %10d   %.4f   %.4f   %.4f" % (nm, len(a), a.mean(), *np.percentile(a, [50, 90])))
allact = np.concatenate([f, b]); lab = np.concatenate([np.ones_like(f), np.zeros_like(b)])
order = np.argsort(-allact)
print()
print("  ranking UNCLAIMED patches by activity, precision at the top:")
for fr in (0.05, 0.10, 0.20):
    k = max(1, int(fr*len(allact)))
    print("    top %3.0f%%:  %5.2f%% true foreground   (base rate %.2f%%)"
          % (100*fr, 100*lab[order[:k]].mean(), 100*lab.mean()))
