#!/usr/bin/env python3
"""Change C: train the head to agree with itself across rollout seeds.

A and B opened the direction -- nine of nine metric cells up across three datasets. What
still limits it is the objective. Across five arms on CLEVR, validation RGB MSE ranked
exactly inverse to FG-ARI, so RGB reconstruction points away from grouping; it only helps
when `tv` or `spike` supplies the direction.

Seed invariance is a signal with no such defect. The rollout is stochastic in its inference
seed, so one image under two seeds gives two traces of the same scene. A grouping that
tracks real objects is the same under both; one that tracks rollout noise is not. There is
no colour-reconstruction shortcut: nothing rewards painting a smooth field.

The loss is symmetric KL between the two seeds' soft assignments, after matching candidate
sets -- both seeds share the same `regions` input, so the candidate index spaces align.

That term alone collapses, and the first run of this file showed it: validation FG-ARI fell
to 0.4954 on CLEVR and 0.6817 on Tetrominoes, identically for every arm whatever its rgb and
tv weights. The reason is that agreement has a trivial minimum. Assign every node to one
candidate and the two seeds agree perfectly, so the KL reaches zero while the partition
carries no information. rgb and tv do not resist it; they are optimised by a smooth field,
which is what collapse produces.

So the objective needs a term that makes collapse expensive. The marginal over candidates,
averaged across nodes, is what collapse destroys -- it becomes a point mass. Two guards are
tried against the unguarded control: `ent` maximises that marginal's entropy, and `size`
holds it near the 253 partition's own size distribution, KL(size253 || marginal), which says
reassign nodes if you must but do not change how much of the image each region owns.

Selection also now includes epoch 0, the head before any update. If no epoch beats it, the
honest answer is that C does not help, and change B's rule returns the untrained head rather
than the least bad trained one.
"""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--dataset', required=True)
ap.add_argument('--train-cache', required=True); ap.add_argument('--val-cache', required=True)
ap.add_argument('--eval-cache', required=True); ap.add_argument('--eval-split', required=True)
ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--out', required=True)
ap.add_argument('--val-every', type=int, default=1)
a = ap.parse_args()
CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
from mod_benchmark import common as c
from neutral_readout.head import NeutralRegionHead, candidates

GRID, NODES = 64, 4096
# weight on the seed-agreement term; rgb keeps the decoder honest, tv the boundaries
ARMS = {'plain':        dict(radius=1, rgb=0., tv=0.,  seed=1.0, ent=0., size=0.),
        'ent':          dict(radius=1, rgb=0., tv=0.,  seed=1.0, ent=1., size=0.),
        'size':         dict(radius=1, rgb=0., tv=0.,  seed=1.0, ent=0., size=1.),
        'size_rgb_tv':  dict(radius=1, rgb=1., tv=.01, seed=1.0, ent=0., size=1.),
        'size_weak':    dict(radius=1, rgb=0., tv=0.,  seed=0.1, ent=0., size=1.),
        'size_r2':      dict(radius=2, rgb=0., tv=0.,  seed=1.0, ent=0., size=1.)}


class ContextAdapter(nn.Module):
    def __init__(self, seed, dilation=1):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(262000+seed)
            self.net = nn.Sequential(nn.Conv2d(8,16,3,padding=dilation,dilation=dilation), nn.SiLU(),
                                     nn.Conv2d(16,32,3,padding=dilation,dilation=dilation,bias=False))
            nn.init.zeros_(self.net[-1].weight)
    def forward(self, g):
        return self.net(g.reshape(GRID,GRID,8).permute(2,0,1)[None])[0].permute(1,2,0).reshape(NODES,32)


class Readout(nn.Module):
    def __init__(self, seed, arm):
        super().__init__(); self.cfg = ARMS[arm]
        base = NeutralRegionHead(seed=259000+seed); self.head = base
        self.head.prior_scale.requires_grad_(False); self.head.residual_scale.requires_grad_(False)
        self.content_encoder = copy.deepcopy(base.embedding).requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(259100+seed)
            self.decoder = nn.Sequential(nn.Linear(34,64), nn.SiLU(), nn.Linear(64,64), nn.SiLU(),
                                         nn.Linear(64,3), nn.Sigmoid())
        self.fine = ContextAdapter(seed)
        axis = (torch.arange(GRID, dtype=torch.float32)+.5)/(GRID/2)-1
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('xy', torch.stack([xx.flatten(), yy.flatten()], -1))

    def logits_from(self, feats, gamma, masks, prior, allowed):
        z = F.normalize(self.head.embedding(feats) + self.fine(gamma), dim=-1)
        proto = F.normalize(masks @ z / masks.sum(1,keepdim=True).clamp_min(1), dim=-1)
        sim = proto @ z.T; w = allowed.float()
        mean = (sim*w).sum(0)/w.sum(0).clamp_min(1)
        std = (((sim-mean).square()*w).sum(0)/w.sum(0).clamp_min(1)).clamp_min(.0025).sqrt()
        return (prior + 2.*((sim-mean)/std).tanh()).masked_fill(~allowed, -1e4)

    def logits(self, r):
        return self.logits_from(r['features'], r['gamma'], r['masks'], r['prior'],
                                r[f"allowed{self.cfg['radius']}"])

    def forward(self, r):
        logits = self.logits(r); p = (logits/.5).softmax(0)
        loss = torch.zeros((), device=logits.device)
        if self.cfg['rgb']:
            with torch.no_grad(): content = self.content_encoder(r['features'])
            latent = (p @ content)/p.sum(1,keepdim=True).clamp_min(1e-6)
            dec = self.decoder(torch.cat([latent[:,None].expand(-1,NODES,-1),
                                          self.xy[None].expand(len(p),-1,-1)],-1))
            loss = loss + self.cfg['rgb']*F.mse_loss((p[:,:,None]*dec).sum(0), r['rgb'])
        if self.cfg['tv']:
            grid = p.reshape(-1,GRID,GRID); tgt = r['rgb'].reshape(GRID,GRID,3); tv = []
            for A,B,ca,cb in [(grid[:,:,1:],grid[:,:,:-1],tgt[:,1:],tgt[:,:-1]),
                              (grid[:,1:,:],grid[:,:-1,:],tgt[1:,:],tgt[:-1,:])]:
                tv.append(((A-B).abs().sum(0)*((ca-cb).square().sum(-1)/.02).neg().exp()).mean())
            loss = loss + self.cfg['tv']*sum(tv)/2
        if self.cfg['ent'] or self.cfg['size']:
            marginal = p.mean(1).clamp_min(1e-8)
            if self.cfg['ent']:
                loss = loss + self.cfg['ent']*(marginal*marginal.log()).sum()
            if self.cfg['size']:
                # KL(size253 || marginal): costly exactly where collapse drops a region
                loss = loss + self.cfg['size']*(r['size_prior']*(r['size_prior'].clamp_min(1e-8).log()
                                                                 - marginal.log())).sum()
        if self.cfg['seed'] and 'features_b' in r:
            # the second seed shares this image's candidate set, so index spaces align
            lb = self.logits_from(r['features_b'], r['gamma_b'], r['masks'], r['prior'],
                                  r[f"allowed{self.cfg['radius']}"])
            la, lbv = (logits/.5).log_softmax(0), (lb/.5).log_softmax(0)
            pa, pb = la.exp(), lbv.exp()
            agree = .5*(F.kl_div(lbv, pa, reduction='batchmean', log_target=False)
                        + F.kl_div(la, pb, reduction='batchmean', log_target=False))
            loss = loss + self.cfg['seed']*agree
        return loss, loss.detach()

    @torch.no_grad()
    def predict(self, r):
        win = self.logits(r).argmax(0); original = r['original']
        out = torch.zeros_like(win); n = len(original)
        out[win < n] = original[win[win < n]]
        extra = win >= n
        if extra.any(): out[extra] = int(original.max()) + 1 + (win[extra] - n)
        return out.reshape(GRID, GRID).cpu().numpy().astype(np.int64)


def load(folder, want_b=False):
    rows = []
    for p in sorted(Path(folder).glob('[0-9]*.pt')):
        d = torch.load(p, map_location='cpu', weights_only=True)
        rgb = d['rgb'].float().div(255.).permute(2,0,1)[None]
        rgb = F.adaptive_avg_pool2d(rgb, GRID)[0].permute(1,2,0).reshape(NODES,3).cuda()
        f = d['features'].cuda(); reg = d['regions_253'].numpy()
        masks, prior = candidates(f, reg)
        lab = torch.as_tensor(reg, device=f.device, dtype=torch.long).flatten()
        row = dict(features=f, gamma=d['gamma'].cuda(), rgb=rgb, regions=reg,
                   masks=masks.float(), prior=prior, original=torch.unique(lab),
                   image_id=int(d['image_id']),
                   size_prior=(masks.float().sum(1)/masks.float().sum()))
        for rad in (1, 2):
            row[f'allowed{rad}'] = F.max_pool2d(masks.float().reshape(-1,1,GRID,GRID),
                                                2*rad+1, stride=1, padding=rad).flatten(1).bool()
        if want_b and 'features_b' in d:
            row['features_b'] = d['features_b'].cuda(); row['gamma_b'] = d['gamma_b'].cuda()
        rows.append(row)
    return rows

t0 = time.time()
train, val, ev = load(a.train_cache, True), load(a.val_cache), load(a.eval_cache)
nb = sum('features_b' in r for r in train)
print(f"{a.dataset}: train {len(train)} ({nb} with a second seed) val {len(val)} eval {len(ev)} "
      f"({time.time()-t0:.0f}s)", flush=True)
assert nb == len(train), f'second seed missing for {len(train)-nb} images'

dataset = Dataset(a.dataset, output); metric = c.benchmark_module('evaluation')
KEYS = ('fg_ari','foreground_iou','matched_object_iou')
def score(rows, preds, split):
    ids = np.array([r['image_id'] for r in rows])
    s = metric.evaluate_patch_masks(torch.from_numpy(np.stack(preds).astype(np.int64)),
                                    torch.from_numpy(dataset.labels(ids, split)))
    return {k: float(s['mean'][k]) for k in KEYS}

base_val = score(val, [r['regions'] for r in val], 'validation')
base_ev = score(ev, [r['regions'] for r in ev], a.eval_split)
print(f"253 baseline   val {base_val['fg_ari']:.4f}   eval {base_ev['fg_ari']:.4f}/"
      f"{base_ev['foreground_iou']:.4f}/{base_ev['matched_object_iou']:.4f}", flush=True)

report = {}
for arm in ARMS:
    torch.manual_seed(1000); m = Readout(0, arm).cuda()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.003)
    best = (-1., None, None); t1 = time.time()
    m.eval(); v0 = score(val, [m.predict(r) for r in val], 'validation')
    best = (v0['fg_ari'], 0, {k: t.detach().cpu().clone() for k, t in m.state_dict().items()})
    for e in range(a.epochs):
        m.train()
        for r in train:
            loss, _ = m(r); opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.); opt.step()
        if (e+1) % a.val_every == 0 or e == a.epochs-1:
            m.eval(); v = score(val, [m.predict(r) for r in val], 'validation')
            if v['fg_ari'] > best[0]:
                best = (v['fg_ari'], e+1, {k: t.detach().cpu().clone() for k, t in m.state_dict().items()})
    report[arm] = dict(val_fg_ari=best[0], epoch=best[1], val_fg_ari_epoch0=v0['fg_ari'])
    print(f"  {arm:14s} val fg_ari {best[0]:.4f} at epoch {best[1]}"
          f"   (epoch 0 was {v0['fg_ari']:.4f})   ({time.time()-t1:.0f}s)", flush=True)
    torch.save(dict(model=best[2], epoch=best[1], val_fg_ari=best[0], arm=arm),
               Path(a.out).parent/f'{a.dataset}_C_{arm}.pt')

pick = max(report, key=lambda k: report[k]['val_fg_ari'])
blob = torch.load(Path(a.out).parent/f'{a.dataset}_C_{pick}.pt', map_location='cuda:0', weights_only=True)
m = Readout(0, pick).cuda().eval(); m.load_state_dict(blob['model'])
preds = [m.predict(r) for r in ev]
head_ev = score(ev, preds, a.eval_split)
moved = float(np.mean([(p != r['regions']).mean() for p, r in zip(preds, ev)]))
print(f"\nselected: {pick} (val {report[pick]['val_fg_ari']:.4f})")
print(f"{'253':10s} {base_ev['fg_ari']:9.4f} {base_ev['foreground_iou']:9.4f} {base_ev['matched_object_iou']:9.4f}")
print(f"{'head C':10s} {head_ev['fg_ari']:9.4f} {head_ev['foreground_iou']:9.4f} {head_ev['matched_object_iou']:9.4f}")
print(f"{'delta':10s} {head_ev['fg_ari']-base_ev['fg_ari']:+9.4f} {head_ev['foreground_iou']-base_ev['foreground_iou']:+9.4f} "
      f"{head_ev['matched_object_iou']-base_ev['matched_object_iou']:+9.4f}   changed {moved:.4f}", flush=True)
Path(a.out).write_text(json.dumps(dict(dataset=a.dataset, change='C seed-invariance', arm=pick,
    epoch=blob['epoch'], arms=report, baseline_253=base_ev, head_C=head_ev,
    patches_changed=moved, eval_count=len(ev)), indent=1)+'\n')
print("FIT4 DONE ->", a.out, flush=True)
