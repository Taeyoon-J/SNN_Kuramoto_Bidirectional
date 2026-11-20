#!/usr/bin/env python3
"""Fit the head on a core whose partition is too fragmented for the standard forward.

The Multi-dSprites target-trained core produces 489 candidates per image on average and up to
1,834, against 20 for the CLEVR core on the same images. fit5's decoder materialises
[candidates, 4096, 34] and the backward pass keeps it, which overflows 23 GiB. That overflow is
itself a finding -- the core is so fragmented the readout is infeasible at standard settings --
but a number is still worth having, so the decoder is evaluated in gradient-checkpointed chunks
over the candidate axis.

The arithmetic is unchanged: the RGB prediction is sum_c p[c] * decoder(latent[c], xy), and a
sum splits over chunks exactly. Only the peak memory differs.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit5

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--dataset', required=True)
ap.add_argument('--train-cache', required=True); ap.add_argument('--eval-cache', required=True)
ap.add_argument('--eval-split', required=True); ap.add_argument('--arm', default='local_rgb_trust')
ap.add_argument('--epochs', type=int, default=2); ap.add_argument('--chunk', type=int, default=48)
ap.add_argument('--out', required=True)
a = ap.parse_args()
output = fit5.connect(a)
from full_benchmark.data import Dataset
from mod_benchmark import common as c

GRID, NODES = fit5.GRID, fit5.NODES

class Chunked(fit5.Readout):
    """fit5.Readout with the decoder summed in checkpointed chunks over candidates."""
    def forward(self, r):
        logits = self.logits(r); p = (logits/.5).softmax(0)
        with torch.no_grad(): content = self.content_encoder(r['features'])
        latent = (p @ content)/p.sum(1, keepdim=True).clamp_min(1e-6)
        xy = self.xy
        def piece(lat, prob):
            dec = self.decoder(torch.cat([lat[:, None].expand(-1, NODES, -1),
                                          xy[None].expand(len(lat), -1, -1)], -1))
            return (prob[:, :, None]*dec).sum(0)
        acc = None
        for i in range(0, len(p), a.chunk):
            sl = slice(i, i+a.chunk)
            part = checkpoint(piece, latent[sl], p[sl], use_reentrant=False)
            acc = part if acc is None else acc + part
        # fit5 applies the RGB MSE unconditionally at weight 1, then tv and spike at their
        # arm weights, then anchor. Mirrored exactly; only the decoder's memory differs.
        loss = F.mse_loss(acc, r['rgb'])
        if self.cfg['tv'] or self.cfg['spike']:
            grid = p.reshape(-1, GRID, GRID); tgt = r['rgb'].reshape(GRID, GRID, 3)
            temp = r['temporal']; tv, pair = [], []
            for A, B, ca, cb, ta, tb in [
                    (grid[:, :, 1:], grid[:, :, :-1], tgt[:, 1:], tgt[:, :-1], temp[:, 1:], temp[:, :-1]),
                    (grid[:, 1:, :], grid[:, :-1, :], tgt[1:, :], tgt[:-1, :], temp[1:, :], temp[:-1, :])]:
                edge = ((ca-cb).square().sum(-1)/.02).neg().exp()
                tv.append(((A-B).abs().sum(0)*edge).mean())
                pair.append(F.mse_loss((A*B).sum(0), (ta*tb).sum(-1).clamp_min(0).prod(-1)))
            loss = loss + self.cfg['tv']*sum(tv)/2 + self.cfg['spike']*sum(pair)/2
        if self.cfg['anchor']:
            loss = loss + self.cfg['anchor']*F.cross_entropy(logits.T/.5, r['prior_argmax'])
        return loss, loss.detach()

train, ev = fit5.load(a.train_cache, 'regions_253'), fit5.load(a.eval_cache, 'regions_253')
nc = [int(r['masks'].shape[0]) for r in train]
print(f'{a.dataset}: {len(train)} train, {len(ev)} eval; candidates/image '
      f'mean {np.mean(nc):.1f} max {max(nc)}; chunk {a.chunk}', flush=True)

dataset = Dataset(a.dataset, output); metric = c.benchmark_module('evaluation')
KEYS = ('fg_ari', 'foreground_iou', 'matched_object_iou')
ids = np.array([r['image_id'] for r in ev])
target = torch.from_numpy(dataset.labels(ids, a.eval_split))
def score(preds):
    s = metric.evaluate_patch_masks(torch.from_numpy(np.stack(preds).astype(np.int64)), target)
    return {k: float(s['mean'][k]) for k in KEYS}

base = score([r['regions'] for r in ev])
torch.manual_seed(1000); m = Chunked(0, a.arm).cuda()
opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=.003)
t0 = time.time()
for e in range(a.epochs):
    m.train()
    for r in train:
        loss, _ = m(r); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([q for q in m.parameters() if q.requires_grad], 1.); opt.step()
    print(f'  epoch {e+1}/{a.epochs} done ({time.time()-t0:.0f}s)', flush=True)
m.eval()
preds = [m.predict(r) for r in ev]
head = score(preds)
moved = float(np.mean([(p != r['regions']).mean() for p, r in zip(preds, ev)]))
print(f'\n{a.dataset} ({len(ev)} images, {a.eval_split})')
print(f"{'':16s} {'fg_ari':>9s} {'fg_iou':>9s} {'obj_iou':>9s}")
print(f"{'core':16s} {base['fg_ari']:9.4f} {base['foreground_iou']:9.4f} {base['matched_object_iou']:9.4f}")
print(f"{'+ head':16s} {head['fg_ari']:9.4f} {head['foreground_iou']:9.4f} {head['matched_object_iou']:9.4f}")
print(f"{'delta':16s} {head['fg_ari']-base['fg_ari']:+9.4f} "
      f"{head['foreground_iou']-base['foreground_iou']:+9.4f} "
      f"{head['matched_object_iou']-base['matched_object_iou']:+9.4f}   changed {moved:.4f}", flush=True)
Path(a.out).write_text(json.dumps(dict(dataset=a.dataset, arm=a.arm, epochs=a.epochs,
    candidates_mean=float(np.mean(nc)), candidates_max=int(max(nc)), chunk=a.chunk,
    baseline_253=base, head_recipe=head, patches_changed=moved,
    note='gradient-checkpointed chunked decoder; arithmetic identical to fit5'), indent=1)+'\n')
print('CHUNK DONE ->', a.out, flush=True)
