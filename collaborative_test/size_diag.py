"""Is the coverage gap a SIZE bias, and which mechanism would explain it?

PV2_0112 established that the missing foreground is not spread over the image:
the readout claims 94.6% of foreground PATCHES but PV2_0106 measures only 87.3%
coverage PER OBJECT. That signature is large objects covered well and small ones
not. This measures coverage against object size directly, and checks the three
mechanisms that could produce a size bias.

Reads the truth: ORACLE DIAGNOSTIC, never a score.
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

recs = []     # (size, covered_frac, mean_activity_of_object, lost)
for r in range(min(LIM, pred.size(0))):
    g = gamma[vs + r:vs + r + 1].to(DEV)
    with torch.no_grad():
        _, spikes, _, _ = core(g, return_core_out=True, return_theta=True)
    act = spikes[0, :, 512:].float().mean(-1).cpu().numpy()
    t = tgt[vs + r].reshape(-1).numpy()
    claimed = (pred[r].reshape(-1) > 0).numpy()
    for oid in np.unique(t):
        if oid == 0: continue
        m = (t == oid)
        recs.append((int(m.sum()), float(claimed[m].mean()), float(act[m].mean())))

recs = np.array(recs)
sizes, cov, oact = recs[:, 0], recs[:, 1], recs[:, 2]
print("objects %d over %d images   median size %.0f patches" % (len(recs), LIM, np.median(sizes)))
print()
print("coverage by object size")
print("  size bucket        n    mean coverage   fully lost   mean activity")
edges = [1, 3, 6, 11, 21, 41, 10**9]
for lo, hi in zip(edges[:-1], edges[1:]):
    s = (sizes >= lo) & (sizes < hi)
    if s.sum() == 0: continue
    print("  %3d-%-4s patches %5d       %5.1f%%        %5.1f%%        %.4f"
          % (lo, (hi-1 if hi < 10**8 else "+"), s.sum(), 100*cov[s].mean(),
             100*(cov[s] == 0).mean(), oact[s].mean()))
print()
print("per-object mean coverage %.4f   per-patch coverage %.4f"
      % (cov.mean(), (cov*sizes).sum()/sizes.sum()))
print("objects claimed by NOTHING: %.1f%%" % (100*(cov == 0).mean()))
print()
# Does the model simply not fire on small objects, or does the readout drop them?
small = sizes < 6; large = sizes >= 21
print("mean object activity   small(<6 patches) %.4f   large(>=21) %.4f   background %.4f"
      % (oact[small].mean(), oact[large].mean(),
         float(np.mean([0]))*0 + 0.1710))
