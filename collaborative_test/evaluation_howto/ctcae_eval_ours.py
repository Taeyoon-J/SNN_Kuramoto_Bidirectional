#!/usr/bin/env python3
"""Evaluate a trained CAE / CAE++ / CtCAE checkpoint on our frames, with the ORACLE REMOVED.

Why this script exists. `ctcae/models/ctcae_eval.py:85` reads the number of KMeans clusters off
the ground truth of each image:

    num_clusters_img = int(len(torch.unique(labels_true[img_idx])))
    k_means = KMeans(n_clusters=num_clusters_img, ...)

`syncx/utils/utils_eval.py:159,172` is the same code. Under our evaluation contract that is an
oracle -- "anything that used the true object count or the true masks to pick a cluster count or
a threshold is an oracle result and is reported separately as a diagnostic" -- so a row produced
that way cannot sit next to ours. Note this is a stronger form than AKOrN's, which fixed k to a
per-dataset constant (4/7/11) rather than reading it per image.

This reproduces their feature space exactly -- their own `phase_to_cartesian_coordinates` on
their own `complex_phase` and `clip_and_rescale(complex_magnitude, phase_mask_threshold)` -- and
changes one thing: **k is a fixed per-dataset constant passed in, never read from the truth.**
Several k are written per run so the choice can be made on validation, which is where our
contract requires hyperparameters to be chosen.

The predicted cluster ids are arbitrary, so `bg_any.py` still has to designate a background,
also on validation.

Output: one npz per k, holding `preds` [N,64,64] int32 and `ids` [N] int64.
"""
import argparse, os, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--profile', required=True, help='e.g. ctcae_clevr_64x64')
ap.add_argument('--ckpt', required=True)
ap.add_argument('--dataset', required=True, help='our dataset name: clevr|tetrominoes|multi_dsprites')
ap.add_argument('--split', required=True, help='validation | fixed320')
ap.add_argument('--k', required=True, help='comma-separated cluster counts, e.g. 4,5,7,11')
ap.add_argument('--out-prefix', required=True)
ap.add_argument('--code', default='/work/USERS/tkim1/sw0255b/benchmark_codes')
ap.add_argument('--config', default='/work/USERS/tkim1/sw0255b/benchmark_codes/configs/frontier.json')
ap.add_argument('--run', default='/work/USERS/tkim1/sw0255b/run')
ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--limit', type=int, default=0,
                help='use only the first N frames; validation subsampling for the k sweep, so '
                     'the cost of choosing k does not scale with a 1,000-image validation split')
a = ap.parse_args()

# --- our frames and ids -------------------------------------------------------------------
sys.path.insert(0, str(Path(a.code)/'vendor')); sys.path.insert(0, a.code)
import portable_runtime as pr
cfg = pr.configure(a.config); output = pr.connect_inputs(cfg, a.run)
from full_benchmark.data import Dataset
ds = Dataset(a.dataset, output)
if a.split == 'fixed320':
    ids, src = np.asarray(ds.test)[:320], 'test'
else:
    ids, src = np.asarray(ds.validation), 'validation'
if a.limit:
    ids = ids[:a.limit]
rgb = np.asarray(ds.rgb(ids, src)).astype(np.uint8)        # (N,128,128,3)
print(f'{a.dataset} {a.split}: {len(ids)} frames {rgb.shape} ids {ids[0]}..{ids[-1]}', flush=True)

# --- their model, their args --------------------------------------------------------------
CT = Path('/work/USERS/tkim1/baselines/_clone/ctcae')
os.chdir(CT); sys.path.insert(0, str(CT))
sys.argv = ['train.py', f'--profile={a.profile}', '-use_wandb=0', f'-seed={a.seed}']
import torch
from PIL import Image
import train as ctrain
from models import ctcae_eval

model, _ = ctrain.build_model(ctrain.args, 3)
blob = torch.load(a.ckpt, map_location='cpu', weights_only=False)
state = blob['model_state_dict'] if 'model_state_dict' in blob else blob
model.load_state_dict(state)
model = model.cuda().eval()
res = (64, 64)
print(f'loaded {a.ckpt}; profile {a.profile}; resolution {res}', flush=True)

# their loader's own preprocessing: PIL resize to the model resolution, then /255
def prep(batch_rgb):
    out = np.empty((len(batch_rgb), 3, res[0], res[1]), np.float32)
    for i, img in enumerate(batch_rgb):
        pil = Image.fromarray(img).resize(res)
        out[i] = np.transpose(np.asarray(pil, np.float32), (2, 0, 1)) / 255.
    return torch.from_numpy(out)

# --- forward, collecting their feature space ----------------------------------------------
feats = []
with torch.no_grad():
    for s in range(0, len(rgb), a.batch):
        x = prep(rgb[s:s+a.batch]).cuda()
        o = model(x, 0)
        mag = ctcae_eval.clip_and_rescale(o['complex_magnitude'],
                                          ctrain.args.phase_mask_threshold)
        cart = ctcae_eval.phase_to_cartesian_coordinates(
            ctrain.args.phase_mask_threshold, o['complex_phase'], mag)
        # b p c h w -> b h w (c p), exactly their rearrange before KMeans
        cart = cart.permute(0, 3, 4, 2, 1).reshape(cart.shape[0], res[0], res[1], -1)
        feats.append(cart.detach().cpu().numpy())
        print(f'  {min(s+a.batch, len(rgb))}/{len(rgb)}', flush=True)
feats = np.concatenate(feats, 0)
print(f'features {feats.shape}', flush=True)

# --- KMeans with a FIXED k, never the truth -----------------------------------------------
from sklearn.cluster import KMeans
for k in [int(v) for v in a.k.split(',')]:
    preds = np.zeros((len(feats), res[0], res[1]), np.int32)
    for i, f in enumerate(feats):
        flat = f.reshape(-1, f.shape[-1])
        km = KMeans(n_clusters=k, n_init=10, random_state=a.seed).fit(flat)
        preds[i] = km.labels_.reshape(res)
    f_out = f'{a.out_prefix}_k{k}.npz'
    np.savez_compressed(f_out, preds=preds, ids=ids.astype(np.int64), k=k)
    print(f'k={k}: -> {f_out}  labels 0..{preds.max()}', flush=True)
print('CTCAE_EVAL_DONE', flush=True)
