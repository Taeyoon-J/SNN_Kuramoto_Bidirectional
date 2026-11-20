"""Is there a label-free signal that tells a spurious group from a real one?

PV2_0114 measured that false-positive foreground costs 0.2006 of foreground IoU
against 0.0428 for the patches we miss -- 4.7x. So the lever is deleting groups,
not adding them. The base rate is favourable too: 22.2% of predicted foreground
is wrong, against the 0.73% base rate that killed PV2_0112.

PV2_0112's lesson was that a stage must not be built on an unverified signal.
This measures, per predicted group, every label-free quantity available, against
the fraction of that group that is truly foreground. If nothing separates, no
suppression stage is worth writing.

ORACLE DIAGNOSTIC -- reads the truth to score the signal, never to produce a mask.
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
lab = torch.load("/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt", map_location="cpu")["patch_labels"]
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

rows = []
for r in range(min(LIM, pred.size(0))):
    g = gamma[vs + r:vs + r + 1].to(DEV)
    with torch.no_grad():
        _, spikes, _, _ = core(g, return_core_out=True, return_theta=True)
    x = spikes[0, :, 512:].float()
    act = x.mean(-1)
    xc = x - x.mean(-1, keepdim=True)
    xc = xc / xc.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    C = (xc @ xc.t()).cpu().numpy()                      # spike correlation
    a = act.cpu().numpy()
    p = pred[r].reshape(-1).numpy(); t = (lab[vs + r].reshape(-1) > 0).numpy()
    for gid in np.unique(p):
        if gid == 0: continue
        idx = np.flatnonzero(p == gid)
        if idx.size < 2: continue
        sub = C[np.ix_(idx, idx)]
        coh = (sub.sum() - np.trace(sub)) / (idx.size*(idx.size-1))
        rr, cc = idx // GRID, idx % GRID
        border = float(min(rr.min(), cc.min(), GRID-1-rr.max(), GRID-1-cc.max()) == 0)
        rows.append((idx.size, coh, a[idx].mean(), border, t[idx].mean()))

R = np.array(rows)
size, coh, actm, bord, truefrac = R[:,0], R[:,1], R[:,2], R[:,3], R[:,4]
spurious = truefrac < 0.5
print("predicted groups %d over %d images   spurious (<50%% true fg) %d (%.1f%%)"
      % (len(R), LIM, spurious.sum(), 100*spurious.mean()))
print()
print("signal              real groups      spurious       separation (AUC)")
def auc(s):   # P(spurious has lower s than real)
    from itertools import product
    a, b = s[~spurious], s[spurious]
    if len(a)==0 or len(b)==0: return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([a,b])))
    ra = ranks[:len(a)].sum()
    return (ra - len(a)*(len(a)-1)/2) / (len(a)*len(b))
for nm, s in (("internal coherence", coh), ("mean activity   ", actm),
              ("group size      ", size), ("touches border  ", bord)):
    print("  %s   %8.4f      %8.4f        %.3f" % (nm, s[~spurious].mean(), s[spurious].mean(), auc(s)))
print()
print("  (AUC 0.5 = no signal, 1.0 = spurious always ranks lowest, 0.0 = always highest)")
print()
for nm, s, lo in (("coherence", coh, True), ("activity", actm, True)):
    v = s if lo else -s
    for q in (0.10, 0.20, 0.30):
        thr = np.quantile(v, q)
        sel = v <= thr
        print("  drop the lowest %2.0f%% by %s: %5.1f%% of dropped are spurious "
              "(base %.1f%%), removes %.1f%% of real groups"
              % (100*q, nm, 100*spurious[sel].mean(), 100*spurious.mean(),
                 100*(sel & ~spurious).sum()/max((~spurious).sum(),1)))
