"""Can the peer's merge stage fire on OUR affinity at all?

Their foreground_induced_merge agglomerates on signed_pair_energy, which is

    energy = (1 + NATIVE_NODES * (A + A^T) / (2 * coupling_gain)) * logit(Q)

so a pair merges only when logit(Q) > 0, i.e. Q > 0.5 (the adjacency factor is positive
and only scales the magnitude). Q is the spike affinity, the same quantity our readout
thresholds at 0.05.

If our within-object Q rarely exceeds 0.5, porting the stage buys nothing -- no pair would
ever have positive gain and every foreground patch would stay its own group. That would be
worse than our current readout, not better. Measure before porting.

Reported: the distribution of Q over four-connected foreground pairs, split by whether the
two patches belong to the SAME true object, with the fraction above 0.5 in each.
"""
import sys, json, argparse
import numpy as np, torch
sys.path.insert(0, "/export_home/tkim1/collaborative_test")
sys.path.insert(0, "/export_home/tkim1")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="multi_dsprites")
ap.add_argument("--grid", type=int, default=64)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--limit", type=int, default=16)
a = ap.parse_args()

import evaluate_model as em

M = "/work/USERS/tkim1/mod"
man = json.load(open("%s/split_manifest_%s.json" % (M, a.dataset)))
lo, _ = man["ranges"]["validation"]
tg = torch.load("%s/targets_%s_grid%d.pt" % (M, a.dataset, a.grid),
                map_location="cpu")["patch_labels"].long()[lo:lo + a.limit]
gam = torch.load("%s/gamma_%s_grid%d.pt" % (M, a.dataset, a.grid),
                 map_location="cpu")[lo:lo + a.limit].float()

import types
args = types.SimpleNamespace()    # only the fields build_core and _spike_affinity read
# Everything build_core reads. The first block is what our evaluation command sets
# explicitly; the second is evaluate_model's own declared defaults, copied rather than
# guessed so the probe builds the same model the evaluation does.
for k, v in dict(checkpoint=a.checkpoint, device="cuda", num_regions=a.grid * a.grid,
                 grid=a.grid, settle=512, num_time_steps=1024, spike_per_component=True,
                 geodesic_steps=3, geodesic_contrast=2.0, graph_spatial_decay=0.35,
                 geodesic_chunk=2, membrane_threshold_mode="population",
                 membrane_threshold_k=0.5,
                 osc_dim=4, freq_gain=2.0, gate_mode="raw", graph_top_k=32,
                 geodesic_base=1.0, geodesic_power=1.0, geodesic_radius=1.5,
                 coupling_balance=0.0, dendrite_per_region=False,
                 graph_feedback_strength=0.0, spike_pulse_gain=0.0,
                 no_center_pulse=False, affinity_chunk=0).items():
    setattr(args, k, v)

core = em.build_core(args, "cuda")
with torch.no_grad():
    _, spikes, membrane, theta = core(gam.cuda(), return_core_out=True, return_theta=True)
    comps = core.last_component_spikes
    q = em._spike_affinity(spikes.float().cpu(), comps.float().cpu() if comps is not None else None, args)

print("affinity Q: shape %s  min %.4f  max %.4f  mean %.4f"
      % (tuple(q.shape), q.min(), q.max(), q.mean()))
print()
G = a.grid
same_all, diff_all = [], []
for i in range(q.shape[0]):
    lab = tg[i].reshape(-1)
    fg = (lab > 0).nonzero(as_tuple=True)[0]
    if fg.numel() < 2: continue
    qi = q[i]
    for idx in fg.tolist():
        r, c = divmod(idx, G)
        for dr, dc in ((0, 1), (1, 0)):
            rr, cc = r + dr, c + dc
            if rr >= G or cc >= G: continue
            j = rr * G + cc
            if lab[j] <= 0: continue
            v = float(qi[idx, j])
            (same_all if lab[idx] == lab[j] else diff_all).append(v)

for nm, arr in (("same object", same_all), ("different object", diff_all)):
    x = np.array(arr)
    if x.size == 0: print("%-18s none" % nm); continue
    print("%-18s n=%-7d mean %.4f  median %.4f  p90 %.4f  max %.4f   ABOVE 0.5: %.1f%%"
          % (nm, x.size, x.mean(), np.median(x), np.percentile(x, 90), x.max(),
             100.0 * (x > 0.5).mean()))
print()
print("  the merge fires only on pairs with Q > 0.5.")
print("  it needs same-object pairs above 0.5 and different-object pairs below it.")
