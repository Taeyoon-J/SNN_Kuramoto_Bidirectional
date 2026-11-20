#!/usr/bin/env python3
"""Score a trained checkpoint on the 320 validation images, with the peer's own scoring code.

Why this exists. `evaluation.evaluate` reads `dataset.test` -- for CLEVR the 320 images the
peer labels, in data.py itself, as "fixed320 reserved evaluation, not an independent
original-paper TEST split". Every configuration we compare costs one more read of it. The
320 validation images (ids 1000-1319) have never been scored by anything in this benchmark,
so selection belongs there and fixed320 can be read once, at the end, for the setting we
chose.

How it avoids touching locked code. `evaluate` is called unmodified. The only change is a
shallow copy of the Dataset whose `.test` points at `.validation`. That is safe because
`Dataset.rgb` and `Dataset.labels` index the HDF5 (CLEVR) or the .npy (the others) **by
image id**; the split name only routes CLEVR's 'train' case, which this never takes. So the
predictions, the spike clustering, the merge, the metric module and `c.immutable()` are all
the originals, byte for byte -- see contracts/source_snapshot.json.

The result is written to its own directory and tagged `split: validation`, so it cannot be
confused with a fixed320 number later.
"""
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True, help='the output root the job was trained into')
ap.add_argument('--model', type=int, required=True)
ap.add_argument('--dataset', default='clevr')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default=None, help='default: <run>/jobs/.../validation')
ap.add_argument('--device', default='cuda:0')
a = ap.parse_args()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
import portable_runtime as pr

config = pr.configure(a.config)
output = pr.connect_inputs(config, a.run)

from full_benchmark.data import Dataset
from full_benchmark.model import FreshModel
from full_benchmark import evaluation as ev

job = Path(a.run) / f'jobs/model{a.model}/{a.dataset}/seed{a.seed}'
assert job.is_dir(), f'no such job: {job}'
out = Path(a.out) if a.out else job / 'validation'

resume = torch.load(job / 'resume.pt', map_location=a.device, weights_only=True)
initial = torch.load(job / 'initial.pt', map_location='cpu', weights_only=True)
print(f"checkpoint: updates {resume['updates']}, images_seen {resume['images_seen']}")

dataset = Dataset(a.dataset, output)
model = FreshModel(a.model, a.seed, normalization=initial['normalization']).to(a.device)
model.load_state_dict(resume['model'])
model.eval()

proxy = copy.copy(dataset)
proxy.test = dataset.validation          # the one and only substitution
assert len(proxy.test) == len(dataset.validation)
assert not np.intersect1d(proxy.test, dataset.train).size, 'validation must not be in TRAIN'
assert not np.intersect1d(proxy.test, dataset.test).size, 'validation must not be fixed320'
print(f"scoring {len(proxy.test)} validation images "
      f"(ids {proxy.test.min()}-{proxy.test.max()}), TRAIN was {len(dataset.train)}")

result = ev.evaluate(model, proxy, out)
result.update(split='validation', split_ids=[int(proxy.test.min()), int(proxy.test.max())],
              scored_by='score_validation.py with evaluate() unmodified',
              checkpoint_updates=int(resume['updates']),
              checkpoint_images_seen=int(resume['images_seen']),
              NOT_fixed320=True)
(out / 'result.json').write_text(json.dumps(result, indent=1) + '\n')

print()
print('VALIDATION pixel  ', {k: round(v, 4) for k, v in result['pixel'].items()})
if 'historical_fixed64_patch' in result:
    print('VALIDATION patch64', {k: round(v, 4) for k, v in result['historical_fixed64_patch'].items()})
print('single-object images:', result.get('single_object_images'),
      '| GT used for prediction:', result.get('GT_used_for_prediction'))
print('written to', out / 'result.json')
