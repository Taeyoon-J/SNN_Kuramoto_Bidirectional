#!/usr/bin/env python3
"""Cache frozen-core activities for any dataset/split, so readout variants cost seconds.

Every readout experiment so far re-ran the 1,024-step rollout for each image -- 3.4 s each,
and the readout itself is milliseconds. Caching the activities once turns each later variant
from 20-40 minutes into a few seconds, which is the difference between trying three ideas
and trying thirty.

Stores exactly what the readouts consume: the 144-channel activity features, gamma for the
context branch, and BOTH partitions -- the all-ones one the head has been getting, and the
253 readout's own partition with background, which the next variant needs.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--dataset', required=True)
ap.add_argument('--split', required=True); ap.add_argument('--count', type=int, required=True)
ap.add_argument('--out', required=True); ap.add_argument('--min-group', type=int, default=8)
# --run serves two unrelated purposes: it is the data root for connect_inputs AND it used
# to fix the checkpoint path. Those have to be separable to cache from a core trained on
# something other than clevr/seed0; --job supplies the checkpoint directly and defaults to
# the old location, so existing commands behave exactly as before.
ap.add_argument('--job', default=None,
                help='directory holding resume.pt and initial.pt; '
                     'default <run>/jobs/model253/clevr/seed0')
a = ap.parse_args()
CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
from full_benchmark.model import FreshModel
from readout_benchmark.head import activity_features
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
from full_benchmark.fast_partition import foreground_induced_merge

GRID = 64
job = Path(a.job) if a.job else Path(a.run)/'jobs/model253/clevr/seed0'
if not (job/'resume.pt').exists():
    raise SystemExit(f'no resume.pt under {job}; pass --job for a non-clevr core')
print(f'core checkpoint: {job}', flush=True)
resume = torch.load(job/'resume.pt', map_location='cuda:0', weights_only=True)
initial = torch.load(job/'initial.pt', map_location='cpu', weights_only=True)
model = FreshModel(253, 0, normalization=initial['normalization']).to('cuda:0')
model.load_state_dict(resume['model']); model.eval().requires_grad_(False)
gain = float(model.wrapped.core.graph_generator.log_coupling_gain.exp().detach().cpu())

dataset = Dataset(a.dataset, output)
pool = {'train': dataset.train, 'validation': dataset.validation,
        'test': dataset.test, 'fixed320': dataset.test}[a.split]
ids = np.asarray(pool[:a.count])
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
print(f"{a.dataset} {a.split}: {len(ids)} images -> {out}", flush=True)

t0 = time.time(); done = 0
for n, image_id in enumerate(ids):
    target = out/f'{n:04d}.pt'
    if target.exists(): continue
    rgb = dataset.rgb(np.array([image_id]), a.split)[0]
    seed = 193000+n if a.split in ('test','fixed320') else 251000+int(image_id)
    with torch.no_grad():
        o = model(torch.from_numpy(rgb[None].copy()).cuda(), inference_seed=seed, live_tail_steps=0)
    gm = o['gamma'][0]; gm = gm.T.contiguous() if gm.shape[0] == 8 else gm.contiguous()
    q = o['q'][0].detach().cpu().numpy().astype(np.float64)
    adj = o['adjacency'][0].detach().cpu().numpy().astype(np.float64)
    allones = foreground_induced_merge(q, adj, gain, np.ones((GRID,GRID),dtype=np.int64), grid_shape=(GRID,GRID))['labels']
    sp = o['trace']['spikes'].detach().cpu(); cs = o['trace']['component_spikes'].detach().cpu()
    g = spike_synchrony_components(sp, foreground_threshold=.5, synchrony_threshold=.5,
        min_group_size=a.min_group, settle=512, components=cs, background='largest_component',
        affinity_mode='spike', spatial_grid_size=GRID)
    sup = spatial_components_to_patch_labels(g, GRID, device='cpu').numpy()[0].astype(np.int64)
    with_bg = foreground_induced_merge(q, adj, gain, sup, grid_shape=(GRID,GRID))['labels']
    assert np.array_equal(with_bg > 0, sup > 0)
    tmp = target.with_suffix('.part')
    torch.save(dict(image_id=int(image_id), features=activity_features(o['trace'])[0].cpu(),
                    gamma=gm.cpu(), regions_all=torch.from_numpy(allones),
                    regions_253=torch.from_numpy(with_bg),
                    rgb=torch.from_numpy(np.asarray(rgb).copy())), tmp)
    tmp.replace(target); done += 1; del o, sp, cs
    if done % 20 == 0:
        el = time.time()-t0
        print(f"  {n+1}/{len(ids)}  {el/done:.1f} s/img  eta {(len(ids)-n-1)*el/done/60:.0f} min", flush=True)
(out/'complete.json').write_text(json.dumps(dict(dataset=a.dataset, split=a.split,
    count=len(ids), ids=ids.tolist(), min_group=a.min_group, core_updates=int(resume['updates'])))+'\n')
print("CACHE DONE", flush=True)
