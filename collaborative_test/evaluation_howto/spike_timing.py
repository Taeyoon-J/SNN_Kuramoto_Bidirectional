#!/usr/bin/env python3
"""Does the mask come from spike TIMING, and how different are different objects' spikes?

Three things, on the same rollouts so they are directly comparable.

(1) SEPARABILITY.  Using ground-truth labels only to GROUP patch pairs (never to set any
    threshold), report the affinity Q for same-object pairs, different-object pairs and
    object-background pairs. This answers "how different are each object's spikes".

(2) TIMING ABLATION.  Circularly shift every patch's spike trace by its own random offset
    inside the settled window, leaving the graph, the foreground rule and the whole merge
    untouched. A per-patch shift destroys relative timing across patches while preserving
    each patch's own rate and waveform, so a drop isolates timing.

(3) SPIKE vs GATE.  s_ia = g_i e_ia, and the firing indicator e is 1 in ~99.99% of settled
    steps, so the spike trace is numerically close to the phase gate. Running the identical
    readout on component_gates says whether the result is carried by the spiking layer or
    only by the phase that gates it. This is the honest check on "from spike timing".

Validity check: Q is recomputed here from component_spikes and asserted equal to the model's
own q on the unshifted case, so the ablations differ from the control in one thing only.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--dataset', default='clevr')
ap.add_argument('--split', default='fixed320'); ap.add_argument('--count', type=int, default=40)
ap.add_argument('--steps', type=int, default=1024); ap.add_argument('--min-group', type=int, default=8)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', required=True)
a = ap.parse_args()
CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
from full_benchmark.model import FreshModel
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
from full_benchmark.fast_partition import foreground_induced_merge
from mod_benchmark import common as c
metric = c.benchmark_module('evaluation'); GRID = 64
KEYS = ('fg_ari', 'foreground_iou', 'matched_object_iou')

ds = Dataset(a.dataset, output)
job = Path(a.run)/'jobs/model253/clevr/seed0'
resume = torch.load(job/'resume.pt', map_location='cuda:0', weights_only=True)
initial = torch.load(job/'initial.pt', map_location='cpu', weights_only=True)
model = FreshModel(253, 0, normalization=initial['normalization']).to('cuda:0')
model.load_state_dict(resume['model']); model.eval().requires_grad_(False)
gain = float(model.wrapped.core.graph_generator.log_coupling_gain.exp().detach().cpu())
print(f'core updates {int(resume["updates"])}', flush=True)

pool = {'train': ds.train, 'validation': ds.validation, 'test': ds.test, 'fixed320': ds.test}[a.split]
ids = np.asarray(pool[:a.count])
tgt = torch.from_numpy(ds.labels(ids, a.split)); tn = tgt.numpy()

def affinity(cs, settle):
    """Q_ij = prod_a max(rho_ija, 0) over the settled window. cs is (A, N, T)."""
    w = cs[:, :, settle:].astype(np.float64)
    w = w - w.mean(axis=2, keepdims=True)
    n = np.sqrt((w*w).sum(axis=2, keepdims=True))
    w = w/np.maximum(n, 1e-8)
    Q = np.ones((w.shape[1], w.shape[1]))
    for A in range(w.shape[0]):
        Q *= np.clip(w[A] @ w[A].T, -1, 1).clip(min=0)
    return Q

def readout(q, adj, sp, cs, settle):
    # the classifier wants [B, N, T] and [B, A, N, T]; cs here is (A, N, T), sp is (N, T)
    g = spike_synchrony_components(torch.from_numpy(sp[None]).float(), foreground_threshold=.5,
        synchrony_threshold=.5, min_group_size=a.min_group, settle=settle,
        components=torch.from_numpy(cs[None]).float(), background='largest_component',
        affinity_mode='spike', spatial_grid_size=GRID)
    sup = spatial_components_to_patch_labels(g, GRID, device='cpu').numpy()[0].astype(np.int64)
    return foreground_induced_merge(q, adj, gain, sup, grid_shape=(GRID, GRID))['labels']

rng = np.random.default_rng(a.seed)
preds = {k: [] for k in ('control', 'shift', 'gate')}
sep = {'same': [], 'diff': [], 'obj_bg': []}
qmatch = []
for n, iid in enumerate(ids):
    rgb = ds.rgb(np.array([iid]), a.split)[0]
    with torch.no_grad():
        o = model(torch.from_numpy(rgb[None].copy()).cuda(), inference_seed=193000+n,
                  live_tail_steps=0, total_steps=a.steps)
    st = int(o['settle'])
    q_model = o['q'][0].detach().cpu().numpy().astype(np.float64)
    adj = o['adjacency'][0].detach().cpu().numpy().astype(np.float64)
    cs = o['trace']['component_spikes'].detach().cpu().numpy()[0]          # (4, N, T)
    cg = o['trace']['component_gates'].detach().cpu().numpy()[0]
    del o

    q_mine = affinity(cs, st)
    qmatch.append(float(np.abs(q_mine - q_model).max()))

    # (1) separability, grouped by ground truth -- diagnostic only
    y = tn[n].reshape(-1)
    iu = np.triu_indices(len(y), k=1)
    qi = q_model[iu]; yi, yj = y[iu[0]], y[iu[1]]
    fg = (yi > 0) & (yj > 0)
    sep['same'].append(float(qi[fg & (yi == yj)].mean()))
    sep['diff'].append(float(qi[fg & (yi != yj)].mean()))
    sep['obj_bg'].append(float(qi[((yi > 0) ^ (yj > 0))].mean()))

    # (2) per-patch circular shift inside the settled window
    cs_s = cs.copy()
    off = rng.integers(0, cs.shape[2]-st, size=cs.shape[1])
    tail = cs_s[:, :, st:]
    idx = (np.arange(tail.shape[2])[None, :] + off[:, None]) % tail.shape[2]
    cs_s[:, :, st:] = np.take_along_axis(tail, idx[None, :, :], axis=2)

    preds['control'].append(readout(q_model, adj, cs.mean(0), cs, st))
    preds['shift'].append(readout(affinity(cs_s, st), adj, cs_s.mean(0), cs_s, st))
    preds['gate'].append(readout(affinity(cg, st), adj, cg.mean(0), cg, st))
    if (n+1) % 10 == 0: print(f'  {n+1}/{len(ids)}', flush=True)

res = {'n': len(ids), 'dataset': a.dataset, 'split': a.split,
       'max_abs_q_recompute_minus_model': max(qmatch)}
for k in preds:
    p = torch.from_numpy(np.stack(preds[k]))
    s = metric.evaluate_patch_masks(p, tgt)['mean']
    res[k] = {m: float(s[m]) for m in KEYS}
    res[k]['regions'] = float(np.mean([len(np.unique(x[x > 0])) for x in preds[k]]))
res['separability'] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in sep.items()}
Path(a.out).write_text(json.dumps(res, indent=2) + '\n')
print(json.dumps(res, indent=2))
