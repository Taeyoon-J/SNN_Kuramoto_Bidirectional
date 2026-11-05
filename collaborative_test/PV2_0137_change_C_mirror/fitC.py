#!/usr/bin/env python3
"""Change C, done against an invariance that exists: a horizontal mirror.

The first attempt used the inference seed. PV2_0135 showed the 253 rollout is bitwise
deterministic in it, so that objective compared a tensor with itself and measured nothing.

A mirror is different. The encoder is a learned CNN and is not mirror-equivariant, the coupling
graph carries a learned direction term over ordered node pairs, and the spike clustering then
runs on genuinely different activity -- `cache_flip.py` rolls out `flip(image)` and stores the
result in the mirrored frame. Undo the flip and the two partitions describe the same scene, so
a grouping that follows objects should agree and one that follows an artifact of the rollout
should not. No labels are involved.

The candidate index spaces are kept aligned the same way as before, by giving both sides the
same candidate set: the unmirrored row's `masks`, `prior` and `allowed` are flipped spatially,
which permutes the NODES axis but leaves the candidate axis untouched, so candidate k means the
same object on both sides. The mirrored logits are then flipped back before comparison.

The collapse guards from the previous attempt are kept, because the trivial minimum is still
there: one candidate everywhere agrees with itself under any transform.
"""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit5

GRID, NODES = fit5.GRID, fit5.NODES
ARMS = {'mirror':            dict(radius=1, rgb=0., tv=0.,  mir=1.0, size=1.),
        'mirror_only':       dict(radius=1, rgb=0., tv=0.,  mir=1.0, size=0.),
        'mirror_rgb':        dict(radius=1, rgb=1., tv=0.,  mir=1.0, size=1.),
        'mirror_rgb_tv':     dict(radius=1, rgb=1., tv=.01, mir=1.0, size=1.),
        'mirror_weak_rgb_tv':dict(radius=1, rgb=1., tv=.01, mir=0.1, size=1.),
        'mirror_r2':         dict(radius=2, rgb=1., tv=.01, mir=1.0, size=1.)}

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--dataset', required=True)
ap.add_argument('--train-cache', required=True); ap.add_argument('--val-cache', required=True)
ap.add_argument('--eval-cache', required=True); ap.add_argument('--eval-split', required=True)
ap.add_argument('--regions', default='253'); ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--iou-floor', type=float, default=.01); ap.add_argument('--out', required=True)
a = ap.parse_args()
output = fit5.connect(a)
from full_benchmark.data import Dataset
from mod_benchmark import common as c


def hflip_nodes(t):
    """Mirror a [*, NODES] tensor across the grid's width, leaving every other axis alone."""
    return t.reshape(*t.shape[:-1], GRID, GRID).flip(-1).reshape(*t.shape[:-1], NODES)


def attach_mirror(rows, folder):
    extra = {}
    for f in sorted(Path(folder).glob('[0-9]*.pt')):
        d = torch.load(f, map_location='cpu', weights_only=True)
        if 'features_f' in d:
            extra[int(d['image_id'])] = (d['features_f'], d['gamma_f'])
    kept = 0
    for r in rows:
        if r['image_id'] in extra:
            fb, gb = extra[r['image_id']]
            r['features_f'], r['gamma_f'] = fb.cuda(), gb.cuda()
            # the mirrored run's candidate set is this row's, mirrored: same candidate order
            r['masks_f'] = hflip_nodes(r['masks'])
            r['prior_f'] = hflip_nodes(r['prior'])
            for rad in (1, 2):
                r[f'allowed{rad}_f'] = hflip_nodes(r[f'allowed{rad}'])
            kept += 1
    return kept


class Readout(fit5.Readout):
    def __init__(self, seed, arm):
        super().__init__(seed, 'local_rgb')      # geometry only; cfg replaced below
        self.cfg = ARMS[arm]

    def logits_on(self, feats, gamma, masks, prior, allowed):
        z = F.normalize(self.head.embedding(feats) + self.fine(gamma), dim=-1)
        proto = F.normalize(masks @ z / masks.sum(1, keepdim=True).clamp_min(1), dim=-1)
        sim = proto @ z.T; w = allowed.float()
        mean = (sim*w).sum(0)/w.sum(0).clamp_min(1)
        std = (((sim-mean).square()*w).sum(0)/w.sum(0).clamp_min(1)).clamp_min(.0025).sqrt()
        return (prior + 2.*((sim-mean)/std).tanh()).masked_fill(~allowed, -1e4)

    def logits(self, r):
        return self.logits_on(r['features'], r['gamma'], r['masks'], r['prior'],
                              r[f"allowed{self.cfg['radius']}"])

    def forward(self, r):
        logits = self.logits(r); p = (logits/.5).softmax(0)
        loss = torch.zeros((), device=logits.device)
        if self.cfg['rgb']:
            with torch.no_grad(): content = self.content_encoder(r['features'])
            latent = (p @ content)/p.sum(1, keepdim=True).clamp_min(1e-6)
            dec = self.decoder(torch.cat([latent[:, None].expand(-1, NODES, -1),
                                          self.xy[None].expand(len(p), -1, -1)], -1))
            loss = loss + self.cfg['rgb']*F.mse_loss((p[:, :, None]*dec).sum(0), r['rgb'])
        if self.cfg['tv']:
            grid = p.reshape(-1, GRID, GRID); tgt = r['rgb'].reshape(GRID, GRID, 3); tv = []
            for A, B, ca, cb in [(grid[:, :, 1:], grid[:, :, :-1], tgt[:, 1:], tgt[:, :-1]),
                                 (grid[:, 1:, :], grid[:, :-1, :], tgt[1:, :], tgt[:-1, :])]:
                tv.append(((A-B).abs().sum(0)*((ca-cb).square().sum(-1)/.02).neg().exp()).mean())
            loss = loss + self.cfg['tv']*sum(tv)/2
        if self.cfg['size']:
            marginal = p.mean(1).clamp_min(1e-8)
            loss = loss + self.cfg['size']*(r['size_prior']*(r['size_prior'].clamp_min(1e-8).log()
                                                             - marginal.log())).sum()
        if self.cfg['mir'] and 'features_f' in r:
            lf = self.logits_on(r['features_f'], r['gamma_f'], r['masks_f'], r['prior_f'],
                                r[f"allowed{self.cfg['radius']}_f"])
            lf = hflip_nodes(lf)                      # back into the original frame
            la, lb = (logits/.5).log_softmax(0), (lf/.5).log_softmax(0)
            agree = .5*(F.kl_div(lb, la.exp(), reduction='batchmean')
                        + F.kl_div(la, lb.exp(), reduction='batchmean'))
            loss = loss + self.cfg['mir']*agree
        return loss, loss.detach()


key = 'regions_253' if a.regions == '253' else 'regions_all'
train, val, ev = (fit5.load(a.train_cache, key), fit5.load(a.val_cache, key),
                  fit5.load(a.eval_cache, key))
for r in train:
    r['size_prior'] = r['masks'].sum(1)/r['masks'].sum()
nf = attach_mirror(train, a.train_cache)
print(f"{a.dataset}: train {len(train)} ({nf} mirrored) val {len(val)} eval {len(ev)}", flush=True)
assert nf == len(train), f'mirror missing for {len(train)-nf} images'

dataset = Dataset(a.dataset, output); metric = c.benchmark_module('evaluation')
KEYS = ('fg_ari', 'foreground_iou', 'matched_object_iou')
def score(rows, preds, split):
    ids = np.array([r['image_id'] for r in rows])
    s = metric.evaluate_patch_masks(torch.from_numpy(np.stack(preds).astype(np.int64)),
                                    torch.from_numpy(dataset.labels(ids, split)))
    return {k: float(s['mean'][k]) for k in KEYS}

base_val = score(val, [r['regions'] for r in val], 'validation')
base_ev = score(ev, [r['regions'] for r in ev], a.eval_split)
print(f"253   val {base_val['fg_ari']:.4f}/{base_val['foreground_iou']:.4f}/{base_val['matched_object_iou']:.4f}"
      f"   eval {base_ev['fg_ari']:.4f}/{base_ev['foreground_iou']:.4f}/{base_ev['matched_object_iou']:.4f}",
      flush=True)

floor = base_val['foreground_iou'] - a.iou_floor
report = {}
for arm in ARMS:
    torch.manual_seed(1000); m = Readout(0, arm).cuda()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.003)
    curve = []; states = {}; t1 = time.time()
    for e in range(a.epochs+1):
        if e:
            m.train()
            for r in train:
                loss, _ = m(r); opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.)
                opt.step()
        m.eval(); v = score(val, [m.predict(r) for r in val], 'validation')
        curve.append(dict(epoch=e, **v))
        states[e] = {k: t.detach().cpu().clone() for k, t in m.state_dict().items()}
    allowed = [c_ for c_ in curve if c_['foreground_iou'] >= floor]
    sel = max(allowed or curve, key=lambda c_: c_['fg_ari'])
    report[arm] = dict(curve=curve, selected=sel, candidates_allowed=len(allowed))
    torch.save(dict(model=states[sel['epoch']], arm=arm, selected=sel, curve=curve),
               Path(a.out).parent/f'{a.dataset}_C_{arm}.pt')
    print(f"  {arm:20s} B' epoch {sel['epoch']:2d} val {sel['fg_ari']:.4f}/"
          f"{sel['foreground_iou']:.4f}/{sel['matched_object_iou']:.4f}"
          f"   {len(allowed)}/{len(curve)} pass the floor   ({time.time()-t1:.0f}s)", flush=True)

pick = max(report, key=lambda k: report[k]['selected']['fg_ari'])
blob = torch.load(Path(a.out).parent/f'{a.dataset}_C_{pick}.pt', map_location='cuda:0', weights_only=True)
m = Readout(0, pick).cuda().eval(); m.load_state_dict(blob['model'])
preds = [m.predict(r) for r in ev]
head_ev = score(ev, preds, a.eval_split)
moved = float(np.mean([(p != r['regions']).mean() for p, r in zip(preds, ev)]))
print(f"\nselected by B': {pick} epoch {report[pick]['selected']['epoch']}")
print(f"{'':12s} {'fg_ari':>9s} {'fg_iou':>9s} {'obj_iou':>9s}")
print(f"{'253':12s} {base_ev['fg_ari']:9.4f} {base_ev['foreground_iou']:9.4f} {base_ev['matched_object_iou']:9.4f}")
print(f"{'head C':12s} {head_ev['fg_ari']:9.4f} {head_ev['foreground_iou']:9.4f} {head_ev['matched_object_iou']:9.4f}")
print(f"{'delta':12s} {head_ev['fg_ari']-base_ev['fg_ari']:+9.4f} "
      f"{head_ev['foreground_iou']-base_ev['foreground_iou']:+9.4f} "
      f"{head_ev['matched_object_iou']-base_ev['matched_object_iou']:+9.4f}   changed {moved:.4f}", flush=True)
Path(a.out).write_text(json.dumps(dict(dataset=a.dataset, change='C mirror invariance',
    arm=pick, epoch=report[pick]['selected']['epoch'], arms=report,
    baseline_253_validation=base_val, baseline_253=base_ev, head_C=head_ev,
    patches_changed=moved, eval_count=len(ev)), indent=1)+'\n')
print('FITC DONE ->', a.out, flush=True)
