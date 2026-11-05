#!/usr/bin/env python3
"""A+B with the candidate set precomputed once per image.

`candidates(features, regions)` has no learned parameters -- it clusters detached features
and returns the mask set and prior. Calling it inside every forward made each epoch rerun a
12-iteration farthest-point search over 4096x144 for every image, which is what made the
previous driver spend minutes per arm. Precomputing it per image turns the per-epoch cost
into a few matmuls.

A  the head refines 253's partition (background is a real candidate) instead of the
   all-ones one, where background could never be chosen and Foreground IoU was pinned to
   the truth foreground fraction on every dataset.
B  arms are selected by validation FG-ARI. On CLEVR the five arms' RGB MSE ranked exactly
   opposite to FG-ARI, so RGB-MSE selection picked the worst of them.

B' corrects B. Selecting on FG-ARI alone has no opinion about how much of the image is
   called foreground, and change C found what that costs: on Tetrominoes it chose a head
   scoring +0.0010 FG-ARI and -0.1141 Foreground IoU, because nothing in the rule objected.
   FG-ARI may rise at some cost in Foreground IoU, but a tenth of it is not a small cost.
   So B' takes the highest validation FG-ARI among the (arm, epoch) pairs whose validation
   Foreground IoU is within `--iou-floor` of 253's, and epoch 0 -- the head before any
   update -- is one of the candidates, so the rule can decline to move at all.

   The floor is a stated tolerance, not a tuned quantity, and it is applied on validation.
   Both rules' choices are recorded, so the cost of the correction is visible.
"""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def parse():
  ap = argparse.ArgumentParser()
  ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
  ap.add_argument('--run', required=True); ap.add_argument('--dataset', required=True)
  ap.add_argument('--train-cache', required=True); ap.add_argument('--val-cache', required=True)
  ap.add_argument('--eval-cache', required=True); ap.add_argument('--eval-split', required=True)
  ap.add_argument('--regions', choices=['all','253'], default='253')
  ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--out', required=True)
  ap.add_argument('--val-every', type=int, default=1)
  ap.add_argument('--iou-floor', type=float, default=.01,
                  help="how far below 253's validation Foreground IoU a candidate may fall")
  return ap.parse_args()

GRID, NODES = 64, 4096


def connect(a):
    """Put the benchmark package on the path and open the data root. Importing this module
    must not do it, so that the transfer test can reuse `Readout` without running anything."""
    CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
    import portable_runtime as pr
    config = pr.configure(a.config)
    return pr.connect_inputs(config, a.run)


ARMS = {'local_rgb': dict(radius=1, tv=0., spike=0., anchor=0.),
        'local_rgb_tv': dict(radius=1, tv=.01, spike=0., anchor=0.),
        'local_rgb_spike': dict(radius=1, tv=0., spike=.01, anchor=0.),
        'local_rgb_trust': dict(radius=1, tv=0., spike=.01, anchor=.002),
        'local_rgb_r2': dict(radius=2, tv=.01, spike=.01, anchor=.002)}


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
        # imported here, not at module scope: the package is only on the path after connect()
        from neutral_readout.head import NeutralRegionHead
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

    def logits(self, r):
        masks, prior, allowed = r['masks'], r['prior'], r[f"allowed{self.cfg['radius']}"]
        z = F.normalize(self.head.embedding(r['features']) + self.fine(r['gamma']), dim=-1)
        proto = F.normalize(masks @ z / masks.sum(1,keepdim=True).clamp_min(1), dim=-1)
        sim = proto @ z.T; w = allowed.float()
        mean = (sim*w).sum(0)/w.sum(0).clamp_min(1)
        std = (((sim-mean).square()*w).sum(0)/w.sum(0).clamp_min(1)).clamp_min(.0025).sqrt()
        return (prior + 2.*((sim-mean)/std).tanh()).masked_fill(~allowed, -1e4)

    def forward(self, r):
        logits = self.logits(r); p = (logits/.5).softmax(0)
        with torch.no_grad(): content = self.content_encoder(r['features'])
        latent = (p @ content)/p.sum(1,keepdim=True).clamp_min(1e-6)
        dec = self.decoder(torch.cat([latent[:,None].expand(-1,NODES,-1), self.xy[None].expand(len(p),-1,-1)],-1))
        rgb = (p[:,:,None]*dec).sum(0); mse = F.mse_loss(rgb, r['rgb'])
        loss = mse
        if self.cfg['tv'] or self.cfg['spike']:
            grid = p.reshape(-1,GRID,GRID); tgt = r['rgb'].reshape(GRID,GRID,3); temp = r['temporal']
            tv, pair = [], []
            for A,B,ca,cb,ta,tb in [(grid[:,:,1:],grid[:,:,:-1],tgt[:,1:],tgt[:,:-1],temp[:,1:],temp[:,:-1]),
                                    (grid[:,1:,:],grid[:,:-1,:],tgt[1:,:],tgt[:-1,:],temp[1:,:],temp[:-1,:])]:
                edge = ((ca-cb).square().sum(-1)/.02).neg().exp()
                tv.append(((A-B).abs().sum(0)*edge).mean())
                pair.append(F.mse_loss((A*B).sum(0), (ta*tb).sum(-1).clamp_min(0).prod(-1)))
            loss = loss + self.cfg['tv']*sum(tv)/2 + self.cfg['spike']*sum(pair)/2
        if self.cfg['anchor']:
            loss = loss + self.cfg['anchor']*F.cross_entropy(logits.T/.5, r['prior_argmax'])
        return loss, mse

    @torch.no_grad()
    def predict(self, r):
        win = self.logits(r).argmax(0); original = r['original']
        out = torch.zeros_like(win); n = len(original)
        out[win < n] = original[win[win < n]]
        extra = win >= n
        if extra.any(): out[extra] = int(original.max()) + 1 + (win[extra] - n)
        return out.reshape(GRID, GRID).cpu().numpy().astype(np.int64)


def load(folder, key):
    from neutral_readout.head import candidates
    rows = []
    for p in sorted(Path(folder).glob('[0-9]*.pt')):
        d = torch.load(p, map_location='cpu', weights_only=True)
        rgb = d['rgb'].float().div(255.).permute(2,0,1)[None]
        rgb = F.adaptive_avg_pool2d(rgb, GRID)[0].permute(1,2,0).reshape(NODES,3).cuda()
        f = d['features'].cuda()
        reg = (d[key] if key in d else d['regions']).numpy()
        masks, prior = candidates(f, reg)              # no learned parameters: compute once
        lab = torch.as_tensor(reg, device=f.device, dtype=torch.long).flatten()
        row = dict(features=f, gamma=d['gamma'].cuda(), rgb=rgb, regions=reg,
                   masks=masks.float(), prior=prior, original=torch.unique(lab),
                   prior_argmax=prior.argmax(0), image_id=int(d['image_id']))
        for rad in (1, 2):
            row[f'allowed{rad}'] = F.max_pool2d(masks.float().reshape(-1,1,GRID,GRID),
                                                2*rad+1, stride=1, padding=rad).flatten(1).bool()
        t = f[:, :128].reshape(GRID,GRID,4,32)
        row['temporal'] = F.normalize(t - t.mean(-1, keepdim=True), dim=-1).detach()
        rows.append(row)
    return rows

def main(a, output):
    t0 = time.time()
    key = 'regions_253' if a.regions == '253' else 'regions_all'
    from full_benchmark.data import Dataset
    from mod_benchmark import common as c
    train, val, ev = (load(a.train_cache, key), load(a.val_cache, key),
                      load(a.eval_cache, key))
    print(f"{a.dataset} regions={a.regions}: train {len(train)} val {len(val)} eval {len(ev)} "
          f"(candidates precomputed in {time.time()-t0:.0f}s)", flush=True)

    dataset = Dataset(a.dataset, output); metric = c.benchmark_module('evaluation')
    KEYS = ('fg_ari','foreground_iou','matched_object_iou')
    def score(rows, preds, split):
        ids = np.array([r['image_id'] for r in rows])
        tgt = torch.from_numpy(dataset.labels(ids, split))
        s = metric.evaluate_patch_masks(torch.from_numpy(np.stack(preds).astype(np.int64)), tgt)
        return {k: float(s['mean'][k]) for k in KEYS}

    VS = 'validation'; ES = a.eval_split
    base_val = score(val, [r['regions'] for r in val], VS)
    base_ev = score(ev, [r['regions'] for r in ev], ES)
    print(f"253 baseline   val {base_val['fg_ari']:.4f}/{base_val['foreground_iou']:.4f}/{base_val['matched_object_iou']:.4f}"
          f"   eval {base_ev['fg_ari']:.4f}/{base_ev['foreground_iou']:.4f}/{base_ev['matched_object_iou']:.4f}", flush=True)

    report = {}
    for arm in ARMS:
        torch.manual_seed(1000); m = Readout(0, arm).cuda()
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.003)
        t1 = time.time(); curve = []
        def record(epoch):
            m.eval(); v = score(val, [m.predict(r) for r in val], VS)
            curve.append(dict(epoch=epoch, **v))
            return v
        record(0)                                   # the head before any update is a candidate
        states = {0: {k: t.detach().cpu().clone() for k, t in m.state_dict().items()}}
        for e in range(a.epochs):
            m.train()
            for r in train:
                loss, _ = m(r); opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.); opt.step()
            if (e+1) % a.val_every == 0 or e == a.epochs-1:
                record(e+1)
                states[e+1] = {k: t.detach().cpu().clone() for k, t in m.state_dict().items()}
        floor = base_val['foreground_iou'] - a.iou_floor
        allowed = [c_ for c_ in curve if c_['foreground_iou'] >= floor]
        bprime = max(allowed or curve, key=lambda c_: c_['fg_ari'])
        bonly = max(curve, key=lambda c_: c_['fg_ari'])
        report[arm] = dict(curve=curve, selected=bprime, b_only=bonly,
                           iou_floor=floor, candidates_allowed=len(allowed))
        print(f"  {arm:18s} B' epoch {bprime['epoch']:2d} val {bprime['fg_ari']:.4f}/"
              f"{bprime['foreground_iou']:.4f}/{bprime['matched_object_iou']:.4f}"
              f"   | B alone would take epoch {bonly['epoch']:2d} "
              f"({bonly['fg_ari']:.4f}/{bonly['foreground_iou']:.4f})"
              f"   {len(allowed)}/{len(curve)} pass the floor   ({time.time()-t1:.0f}s)", flush=True)
        torch.save(dict(model=states[bprime['epoch']], arm=arm, regions=a.regions,
                        selected=bprime, curve=curve),
                   Path(a.out).parent/f'{a.dataset}_{a.regions}_{arm}.pt')

    pick = max(report, key=lambda k: report[k]['selected']['fg_ari'])
    blob = torch.load(Path(a.out).parent/f'{a.dataset}_{a.regions}_{pick}.pt',
                      map_location='cuda:0', weights_only=True)
    m = Readout(0, pick).cuda().eval(); m.load_state_dict(blob['model'])
    preds = [m.predict(r) for r in ev]
    head_ev = score(ev, preds, ES)
    moved = float(np.mean([(p != r['regions']).mean() for p, r in zip(preds, ev)]))
    sel = report[pick]['selected']
    print(f"\nselected by B' (validation FG-ARI, Foreground IoU floor "
          f"{report[pick]['iou_floor']:.4f}): {pick} epoch {sel['epoch']}")
    print(f"{'':10s} {'fg_ari':>9s} {'fg_iou':>9s} {'obj_iou':>9s}")
    print(f"{'253':10s} {base_ev['fg_ari']:9.4f} {base_ev['foreground_iou']:9.4f} {base_ev['matched_object_iou']:9.4f}")
    print(f"{'head':10s} {head_ev['fg_ari']:9.4f} {head_ev['foreground_iou']:9.4f} {head_ev['matched_object_iou']:9.4f}")
    print(f"{'delta':10s} {head_ev['fg_ari']-base_ev['fg_ari']:+9.4f} "
          f"{head_ev['foreground_iou']-base_ev['foreground_iou']:+9.4f} "
          f"{head_ev['matched_object_iou']-base_ev['matched_object_iou']:+9.4f}   changed {moved:.4f}", flush=True)
    Path(a.out).write_text(json.dumps(dict(dataset=a.dataset, regions=a.regions, rule="B'",
        iou_floor=a.iou_floor, arm=pick, epoch=sel['epoch'], arms=report,
        baseline_253_validation=base_val, baseline_253=base_ev, head=head_ev,
        patches_changed=moved, eval_count=len(ev)), indent=1)+'\n')
    print("FIT5 DONE ->", a.out, flush=True)


if __name__ == '__main__':
    args = parse()
    main(args, connect(args))
