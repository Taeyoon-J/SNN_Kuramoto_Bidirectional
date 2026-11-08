#!/usr/bin/env python3
"""fit5's B' sweep, with the decoder chunked so it fits a heavily fragmented core.

chain261.sh died here with CUDA OOM on the Multi-dSprites target-trained core: 415 candidates
per image on average and 1,834 at the worst, where fit5 materialises [candidates, 4096, 34] in
the decoder and backward retains it. head_chunk.py already produced the label-free recipe row
this way; this produces the missing B' row so Multi-dSprites has both, like Tetrominoes.

The arithmetic is fit5's exactly -- RGB MSE at weight 1, then tv and spike at their arm weights,
then anchor -- with the decoder's candidate sum evaluated in gradient-checkpointed chunks. A sum
splits over chunks, so only peak memory changes. Selection is rule B' unchanged: highest
validation FG-ARI among (arm, epoch) pairs whose validation Foreground IoU is within --iou-floor
of the core's, epoch 0 included so the rule may decline to move.
"""
import argparse, json, statistics, sys, time
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
ap.add_argument('--train-cache', required=True); ap.add_argument('--val-cache', required=True)
ap.add_argument('--eval-cache', required=True); ap.add_argument('--eval-split', required=True)
ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--iou-floor', type=float, default=.01)
ap.add_argument('--chunk', type=int, default=48); ap.add_argument('--out', required=True)
a = ap.parse_args()
output = fit5.connect(a)
from full_benchmark.data import Dataset
from mod_benchmark import common as c
GRID, NODES = fit5.GRID, fit5.NODES


class Chunked(fit5.Readout):
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


train, val, ev = (fit5.load(a.train_cache, 'regions_253'), fit5.load(a.val_cache, 'regions_253'),
                  fit5.load(a.eval_cache, 'regions_253'))
nc = [int(r['masks'].shape[0]) for r in train]
print(f'{a.dataset}: train {len(train)} val {len(val)} eval {len(ev)}; '
      f'candidates/image mean {np.mean(nc):.1f} max {max(nc)}; chunk {a.chunk}', flush=True)

dataset = Dataset(a.dataset, output); metric = c.benchmark_module('evaluation')
KEYS = ('fg_ari', 'foreground_iou', 'matched_object_iou')
def score(rows, preds, split):
    ids = np.array([r['image_id'] for r in rows])
    s = metric.evaluate_patch_masks(torch.from_numpy(np.stack(preds).astype(np.int64)),
                                    torch.from_numpy(dataset.labels(ids, split)))
    return {k: float(s['mean'][k]) for k in KEYS}

base_val = score(val, [r['regions'] for r in val], 'validation')
base_ev = score(ev, [r['regions'] for r in ev], a.eval_split)
print(f"core   val {base_val['fg_ari']:.4f}/{base_val['foreground_iou']:.4f}/{base_val['matched_object_iou']:.4f}"
      f"   eval {base_ev['fg_ari']:.4f}/{base_ev['foreground_iou']:.4f}/{base_ev['matched_object_iou']:.4f}",
      flush=True)

floor = base_val['foreground_iou'] - a.iou_floor
report = {}
for arm in fit5.ARMS:
    torch.manual_seed(1000); m = Chunked(0, arm).cuda()
    opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=.003)
    curve, states, t0 = [], {}, time.time()
    for e in range(a.epochs+1):
        if e:
            m.train()
            for r in train:
                loss, _ = m(r); opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_([q for q in m.parameters() if q.requires_grad], 1.)
                opt.step()
        m.eval()
        v = score(val, [m.predict(r) for r in val], 'validation')
        curve.append(dict(epoch=e, **v))
        states[e] = {k: t.detach().cpu().clone() for k, t in m.state_dict().items()}
    allowed = [x for x in curve if x['foreground_iou'] >= floor]
    sel = max(allowed or curve, key=lambda x: x['fg_ari'])
    report[arm] = dict(curve=curve, selected=sel, candidates_allowed=len(allowed))
    torch.save(dict(model=states[sel['epoch']], arm=arm, selected=sel, curve=curve),
               Path(a.out).parent/f'{a.dataset}_C_{arm}.pt')
    print(f"  {arm:18s} B' epoch {sel['epoch']:2d} val {sel['fg_ari']:.4f}/"
          f"{sel['foreground_iou']:.4f}/{sel['matched_object_iou']:.4f}"
          f"   {len(allowed)}/{len(curve)} pass the floor   ({time.time()-t0:.0f}s)", flush=True)

pick = max(report, key=lambda k: report[k]['selected']['fg_ari'])
blob = torch.load(Path(a.out).parent/f'{a.dataset}_C_{pick}.pt', map_location='cuda:0', weights_only=True)
m = Chunked(0, pick).cuda().eval(); m.load_state_dict(blob['model'])
preds = [m.predict(r) for r in ev]
head_ev = score(ev, preds, a.eval_split)
moved = float(np.mean([(p != r['regions']).mean() for p, r in zip(preds, ev)]))
print(f"\nselected by B' (floor {floor:.4f}): {pick} epoch {report[pick]['selected']['epoch']}")
print(f"{'':10s} {'fg_ari':>9s} {'fg_iou':>9s} {'obj_iou':>9s}")
print(f"{'core':10s} {base_ev['fg_ari']:9.4f} {base_ev['foreground_iou']:9.4f} {base_ev['matched_object_iou']:9.4f}")
print(f"{'+ head':10s} {head_ev['fg_ari']:9.4f} {head_ev['foreground_iou']:9.4f} {head_ev['matched_object_iou']:9.4f}")
print(f"{'delta':10s} {head_ev['fg_ari']-base_ev['fg_ari']:+9.4f} "
      f"{head_ev['foreground_iou']-base_ev['foreground_iou']:+9.4f} "
      f"{head_ev['matched_object_iou']-base_ev['matched_object_iou']:+9.4f}   changed {moved:.4f}", flush=True)
Path(a.out).write_text(json.dumps(dict(dataset=a.dataset, rule="B'", iou_floor=a.iou_floor,
    arm=pick, epoch=report[pick]['selected']['epoch'], chunk=a.chunk,
    candidates_mean=float(np.mean(nc)), candidates_max=int(max(nc)), arms=report,
    baseline_253_validation=base_val, baseline_253=base_ev, head=head_ev,
    patches_changed=moved, eval_count=len(ev),
    note='gradient-checkpointed chunked decoder; arithmetic identical to fit5'), indent=1)+'\n')
print('FIT5CHUNK DONE ->', a.out, flush=True)
