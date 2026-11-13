#!/usr/bin/env python3
"""Can the mask be read from actual threshold crossings rather than the gated trace?

`PV2_0159` showed the reported readout is numerically a phase readout: with s = g*e and the
firing indicator e equal to 1 in 99.992% of settled steps, running the identical readout on the
gates reproduces every mask to 16 significant figures. CLAUDE.md requires the final mask to come
from a classifier on spikes or membrane and forbids a phase readout, so that has to be resolved
rather than only recorded.

The resolution attempt: raise the firing threshold at INFERENCE ONLY so that e becomes sparse,
and build the affinity from the binary events e themselves. Nothing is retrained; V is read from
the trace the normal rollout already produces.

The foreground set is taken from the control run and held fixed, so the only thing that changes
is the signal the grouping is computed from. `spike_synchrony_components` takes the foreground
activity and the affinity components as separate arguments, which makes that isolation exact.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--dataset', default='clevr')
ap.add_argument('--split', default='fixed320'); ap.add_argument('--count', type=int, default=20)
ap.add_argument('--steps', type=int, default=1024); ap.add_argument('--min-group', type=int, default=8)
ap.add_argument('--mults', default='1.0,1.5,2.0,3.0,5.0')
ap.add_argument('--out', required=True)
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

# the firing threshold the membrane layer actually used, found on the live module
vth = None
for name, mod in model.named_modules():
    for attr in ('v_th', 'vth'):
        v = getattr(mod, attr, None)
        if v is None: continue
        v = float(v.flatten()[0]) if torch.is_tensor(v) else float(v)
        if v > 0: vth = v; print(f'firing threshold from {name}.{attr} = {vth:.6g}'); break
    if vth is not None: break
if vth is None: raise SystemExit('could not find the firing threshold on the model')

pool = {'train': ds.train, 'validation': ds.validation,
        'test': ds.test, 'fixed320': ds.test}[a.split]
ids = np.asarray(pool[:a.count])
tgt = torch.from_numpy(ds.labels(ids, a.split))

def affinity(x, settle):
    """Q from a (A, N, T) trace over the settled window."""
    w = x[:, :, settle:].astype(np.float64)
    w = w - w.mean(axis=2, keepdims=True)
    w = w/np.maximum(np.sqrt((w*w).sum(axis=2, keepdims=True)), 1e-8)
    Q = np.ones((w.shape[1], w.shape[1]))
    for A in range(w.shape[0]):
        Q *= np.clip(w[A] @ w[A].T, -1, 1).clip(min=0)
    return Q

def readout(q, adj, sp, cs, settle):
    g = spike_synchrony_components(torch.from_numpy(sp[None]).float(), foreground_threshold=.5,
        synchrony_threshold=.5, min_group_size=a.min_group, settle=settle,
        components=torch.from_numpy(cs[None]).float(), background='largest_component',
        affinity_mode='spike', spatial_grid_size=GRID)
    sup = spatial_components_to_patch_labels(g, GRID, device='cpu').numpy()[0].astype(np.int64)
    return foreground_induced_merge(q, adj, gain, sup, grid_shape=(GRID, GRID))['labels']

mults = [float(x) for x in a.mults.split(',')]
preds = {('spikes', m): [] for m in mults}
preds[('control', 0)] = []
rates = {m: [] for m in mults}
for n, iid in enumerate(ids):
    rgb = ds.rgb(np.array([iid]), a.split)[0]
    with torch.no_grad():
        o = model(torch.from_numpy(rgb[None].copy()).cuda(), inference_seed=193000+n,
                  live_tail_steps=0, total_steps=a.steps)
    st = int(o['settle'])
    q0 = o['q'][0].detach().cpu().numpy().astype(np.float64)
    adj = o['adjacency'][0].detach().cpu().numpy().astype(np.float64)
    cs = o['trace']['component_spikes'].detach().cpu().numpy()[0]
    V = o['trace']['component_membrane'].detach().cpu().numpy()[0]
    del o
    sp = cs.mean(0)                                   # foreground, held fixed across arms
    preds[('control', 0)].append(readout(q0, adj, sp, cs, st))
    for m in mults:
        # m <= 1 is read as a TARGET FIRING RATE: pick the threshold per component as the
        # (1 - m) quantile of V over the settled window, so the sparsity is what is swept
        # rather than a multiple of a threshold the membrane sits far above. m > 1 keeps the
        # old meaning, a multiple of the trained threshold.
        if m <= 1.0:
            thr = np.quantile(V[:, :, st:].reshape(V.shape[0], -1), 1.0 - m, axis=1)
            e = (V > thr[:, None, None]).astype(np.float32)
        else:
            e = (V > m*vth).astype(np.float32)
        rates[m].append(float(e[:, :, st:].mean()))
        preds[('spikes', m)].append(readout(affinity(e, st), adj, sp, e, st))
    if (n+1) % 5 == 0: print(f'  {n+1}/{len(ids)}', flush=True)

res = {'n': len(ids), 'vth': vth, 'dataset': a.dataset, 'split': a.split, 'arms': {}}
for key, p in preds.items():
    arr = torch.from_numpy(np.stack(p))
    s = metric.evaluate_patch_masks(arr, tgt)['mean']
    if key[0] == 'control': name = 'control (gated trace s)'
    elif key[1] <= 1.0:     name = f'events e, target rate {key[1]:g}'
    else:                   name = f'events e, threshold x{key[1]:g}'
    row = {k: float(s[k]) for k in KEYS}
    row['regions'] = float(np.mean([len(np.unique(x[x > 0])) for x in p]))
    if key[0] == 'spikes': row['firing_rate'] = float(np.mean(rates[key[1]]))
    res['arms'][name] = row
Path(a.out).write_text(json.dumps(res, indent=2) + '\n')
print(json.dumps(res, indent=2))
