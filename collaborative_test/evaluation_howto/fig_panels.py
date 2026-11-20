#!/usr/bin/env python3
"""Dump the four real-data panels for the architecture figure, from the scored 253 core.

The figure's image / spiking-dynamics / spike-correlation / object-mask panels were hand-drawn
schematics. This replaces them with the model's actual output on one CLEVR test image.

Two rules make the panel honest:
  1. The image is chosen as the MEDIAN fg_ari of the 320 reserved ids, not the best. The chosen
     id and its score are written into the npz so the caption can state them.
  2. The rollout is re-run with the same checkpoint, the same seed formula (193000+n) and the
     same T as cache_any.py, and the recomputed regions_253 is COMPARED to the cached one. Any
     divergence is reported in full and both masks are saved, so the figure can show the mask
     that was actually scored rather than a re-run that drifted from it.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--cache', required=True)
ap.add_argument('--dataset', default='clevr'); ap.add_argument('--split', default='fixed320')
ap.add_argument('--out', required=True); ap.add_argument('--steps', type=int, default=1024)
ap.add_argument('--min-group', type=int, default=8)
ap.add_argument('--job', default=None)
ap.add_argument('--pick', default='median', help="median | best | an explicit frame index")
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
metric = c.benchmark_module('evaluation')
GRID = 64

# ---------------------------------------------------------------- 1. choose the frame
dataset = Dataset(a.dataset, output)
files = sorted(Path(a.cache).glob('*.pt'))
frames = []
for f in files:
    d = torch.load(f, map_location='cpu', weights_only=False)
    if 'image_id' not in d: continue
    frames.append((int(f.stem), int(d['image_id']), d['regions_253'].numpy().astype(np.int64)))
ns   = np.array([t[0] for t in frames])
ids  = np.array([t[1] for t in frames], dtype=np.int64)
pred = np.stack([t[2] for t in frames])
order = np.argsort(ids); ns, ids, pred = ns[order], ids[order], pred[order]
target = torch.from_numpy(dataset.labels(ids, a.split))
s = metric.evaluate_patch_masks(torch.from_numpy(pred), target)
fg = s['per_image']['fg_ari'].numpy()
print(f'cache {a.cache}: {len(ids)} frames, mean fg_ari {np.nanmean(fg):.4f}')

n_obj = np.array([len(np.unique(t[t > 0])) for t in target.numpy()])
ok = ~np.isnan(fg)
cand = np.where(ok & (n_obj >= 4) & (n_obj <= 6))[0]
if len(cand) == 0: cand = np.where(ok)[0]
if a.pick == 'median':
    k = cand[np.argsort(fg[cand])[len(cand)//2]]
elif a.pick == 'best':
    k = cand[np.argmax(fg[cand])]
else:
    k = int(np.where(ns == int(a.pick))[0][0])
frame_n, image_id = int(ns[k]), int(ids[k])
print(f'chosen frame {frame_n:04d}  image_id {image_id}  fg_ari {fg[k]:.4f}  '
      f'(median of {len(cand)} 4-6 object scenes; dataset mean {np.nanmean(fg):.4f})  '
      f'truth objects {n_obj[k]}')

# ---------------------------------------------------------------- 2. re-run that frame
job = Path(a.job) if a.job else Path(a.run)/'jobs/model253/clevr/seed0'
resume = torch.load(job/'resume.pt', map_location='cuda:0', weights_only=True)
initial = torch.load(job/'initial.pt', map_location='cpu', weights_only=True)
model = FreshModel(253, 0, normalization=initial['normalization']).to('cuda:0')
model.load_state_dict(resume['model']); model.eval().requires_grad_(False)
gain = float(model.wrapped.core.graph_generator.log_coupling_gain.exp().detach().cpu())

rgb = dataset.rgb(np.array([image_id]), a.split)[0]
seed = 193000 + frame_n
with torch.no_grad():
    o = model(torch.from_numpy(rgb[None].copy()).cuda(), inference_seed=seed,
              live_tail_steps=0, total_steps=a.steps)
q   = o['q'][0].detach().cpu().numpy().astype(np.float64)
adj = o['adjacency'][0].detach().cpu().numpy().astype(np.float64)
sp  = o['trace']['spikes'].detach().cpu()
cs  = o['trace']['component_spikes'].detach().cpu()
print(f"spikes {tuple(sp.shape)} {sp.dtype} | component_spikes {tuple(cs.shape)} | "
      f"q {q.shape} | settle {o['settle']}")

g = spike_synchrony_components(sp, foreground_threshold=.5, synchrony_threshold=.5,
    min_group_size=a.min_group, settle=o['settle'], components=cs,
    background='largest_component', affinity_mode='spike', spatial_grid_size=GRID)
sup = spatial_components_to_patch_labels(g, GRID, device='cpu').numpy()[0].astype(np.int64)
regions = foreground_induced_merge(q, adj, gain, sup, grid_shape=(GRID, GRID))['labels']

cached = pred[k]
same = bool(np.array_equal(regions, cached))
print(f'recomputed regions_253 identical to the cached, scored mask: {same}')
if not same:
    diff = int((regions != cached).sum())
    fg_same = bool(np.array_equal(regions > 0, cached > 0))
    sc = metric.evaluate_patch_masks(torch.from_numpy(regions[None]), target[k][None])
    rec = float(sc['mean']['fg_ari'])
    print(f'  differing patches   : {diff} / {regions.size} ({100.0*diff/regions.size:.2f}%)')
    print(f'  foreground identical: {fg_same}')
    print(f'  region count        : cached {cached.max()}  recomputed {regions.max()}')
    print(f'  fg_ari              : cached {fg[k]:.4f}  recomputed {rec:.4f}')
    print('  -> the figure will use the CACHED mask, which is the one that was scored.')

np.savez_compressed(a.out,
    rgb=np.asarray(rgb), regions_cached=cached, regions_rerun=regions,
    identical=np.bool_(same), truth=target.numpy()[k],
    q=q.astype(np.float32), spikes=sp.numpy().astype(np.float32),
    settle=np.int64(o['settle']), image_id=np.int64(image_id), frame=np.int64(frame_n),
    fg_ari=np.float64(fg[k]), dataset_mean_fg_ari=np.float64(np.nanmean(fg)),
    n_truth_objects=np.int64(n_obj[k]))
print(f'wrote {a.out}')
