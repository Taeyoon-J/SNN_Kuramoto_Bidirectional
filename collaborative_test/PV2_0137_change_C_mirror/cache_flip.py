#!/usr/bin/env python3
"""Cache a rollout of each training image MIRRORED, to give change C an invariance that exists.

PV2_0135 found the 253 rollout is bitwise deterministic in `inference_seed`, so the second-seed
cache held no second seed and change C's objective compared a tensor with itself. Seed
invariance is unavailable for this core. That does not mean no invariance is.

A horizontal mirror is one the core does not satisfy by construction. Its encoder is a learned
CNN, not mirror-equivariant; the coupling graph is built from grid distances and a learned
direction term over ordered node pairs; and the spike clustering then runs on different
activity. So rolling out `flip(image)` genuinely produces a different trace, and after undoing
the flip the two partitions describe the same scene. A grouping that follows objects should
agree; one that follows an artifact of the rollout should not.

Everything is stored in the mirrored frame exactly as the model produced it. The un-flipping is
done by the training driver, where it is visible.

Only TRAIN needs this -- validation and evaluation are scored, not trained on.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--dataset', required=True)
ap.add_argument('--cache', required=True); ap.add_argument('--min-group', type=int, default=8)
a = ap.parse_args()
CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
from full_benchmark.model import FreshModel
from readout_benchmark.head import activity_features
from full_benchmark.fast_partition import foreground_induced_merge
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels

GRID = 64
job = Path(a.run)/'jobs/model253/clevr/seed0'
resume = torch.load(job/'resume.pt', map_location='cuda:0', weights_only=True)
initial = torch.load(job/'initial.pt', map_location='cpu', weights_only=True)
model = FreshModel(253, 0, normalization=initial['normalization']).to('cuda:0')
model.load_state_dict(resume['model']); model.eval().requires_grad_(False)
gain = float(model.wrapped.core.graph_generator.log_coupling_gain.exp().detach().cpu())
dataset = Dataset(a.dataset, output)
folder = Path(a.cache); files = sorted(folder.glob('[0-9]*.pt'))
print(f"{a.dataset}: mirroring {len(files)} cached TRAIN images (min_group {a.min_group})", flush=True)

t0 = time.time(); n = 0
for p in files:
    d = torch.load(p, map_location='cpu', weights_only=True)
    if 'features_f' in d: continue
    image_id = int(d['image_id'])
    rgb = dataset.rgb(np.array([image_id]), 'train')[0]
    mirrored = np.ascontiguousarray(rgb[:, ::-1, :])          # HWC, flip the width axis
    # the same inference seed cache_any.py used for this split, so only the image differs
    with torch.no_grad():
        o = model(torch.from_numpy(mirrored[None].copy()).cuda(),
                  inference_seed=251000 + image_id, live_tail_steps=0)
    gm = o['gamma'][0]; gm = gm.T.contiguous() if gm.shape[0] == 8 else gm.contiguous()
    q = o['q'][0].detach().cpu().numpy().astype(np.float64)
    adj = o['adjacency'][0].detach().cpu().numpy().astype(np.float64)
    sp = o['trace']['spikes'].detach().cpu(); cs = o['trace']['component_spikes'].detach().cpu()
    g = spike_synchrony_components(sp, foreground_threshold=.5, synchrony_threshold=.5,
        min_group_size=a.min_group, settle=512, components=cs, background='largest_component',
        affinity_mode='spike', spatial_grid_size=GRID)
    sup = spatial_components_to_patch_labels(g, GRID, device='cpu').numpy()[0].astype(np.int64)
    with_bg = foreground_induced_merge(q, adj, gain, sup, grid_shape=(GRID, GRID))['labels']
    assert np.array_equal(with_bg > 0, sup > 0)
    d['features_f'] = activity_features(o['trace'])[0].cpu()
    d['gamma_f'] = gm.cpu()
    d['regions_253_f'] = torch.from_numpy(with_bg)
    tmp = p.with_suffix('.part'); torch.save(d, tmp); tmp.replace(p)
    n += 1; del o, sp, cs
    if n % 20 == 0:
        el = time.time()-t0
        print(f'  {n} mirrored in {el/60:.1f} min, {el/n:.1f} s each', flush=True)
print(f'MIRROR DONE for {n} images in {(time.time()-t0)/60:.1f} min', flush=True)
