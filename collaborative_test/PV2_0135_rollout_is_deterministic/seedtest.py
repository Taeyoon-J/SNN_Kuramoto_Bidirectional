#!/usr/bin/env python3
"""Is the 253 rollout actually stochastic in its inference seed?

Change C assumed it is: `cache_seed2.py` re-ran each image at inference_seed = 777000 +
image_id to get a second activity trace. The cache it wrote has features_b and gamma_b
bitwise identical to the first seed, which either means the seed does nothing or means
activity_features discards whatever differs. This checks the trace itself.
"""
import sys
from pathlib import Path
import numpy as np
import torch

B = Path('/work/USERS/tkim1/sw0255b')
sys.path.insert(0, str(B/'benchmark_codes/vendor')); sys.path.insert(0, str(B/'benchmark_codes'))
import portable_runtime as pr
config = pr.configure(str(B/'benchmark_codes/configs/frontier.json'))
output = pr.connect_inputs(config, str(B/'run'))
from full_benchmark.data import Dataset
from full_benchmark.model import FreshModel

job = B/'run/jobs/model253/clevr/seed0'
resume = torch.load(job/'resume.pt', map_location='cuda:0', weights_only=True)
initial = torch.load(job/'initial.pt', map_location='cpu', weights_only=True)
model = FreshModel(253, 0, normalization=initial['normalization']).to('cuda:0')
model.load_state_dict(resume['model']); model.eval().requires_grad_(False)

dataset = Dataset('clevr', output)
image_id = 3
rgb = dataset.rgb(np.array([image_id]), 'train')[0]
x = torch.from_numpy(rgb[None].copy()).cuda()

outs = []
for seed in (0, 777003, 12345):
    with torch.no_grad():
        o = model(x, inference_seed=seed, live_tail_steps=0)
    outs.append({k: o[k] for k in ('gamma', 'q', 'adjacency')} |
                {'spikes': o['trace']['spikes'], 'component_spikes': o['trace']['component_spikes']})
    print(f'seed {seed}: rolled out', flush=True)

names = ['gamma', 'q', 'adjacency', 'spikes', 'component_spikes']
print(f'\nimage {image_id}, three inference seeds, same checkpoint, eval mode:')
for n in names:
    a0 = outs[0][n].float()
    row = []
    for j in (1, 2):
        aj = outs[j][n].float()
        same = torch.equal(outs[0][n], outs[j][n])
        d = float((a0-aj).abs().max())
        row.append(f'vs seed {[0,777003,12345][j]}: identical={same} max|diff|={d:.3e}')
    print(f'  {n:18s} ' + '   '.join(row))
sp = outs[0]['spikes'].float()
print(f'\n  spike tensor {tuple(outs[0]["spikes"].shape)}, mean rate {float(sp.mean()):.4f}, '
      f'total spikes {float(sp.sum()):.0f}')
