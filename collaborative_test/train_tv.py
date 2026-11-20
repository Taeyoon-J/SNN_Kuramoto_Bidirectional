"""Reconstruction plus a total-variation penalty: an UNSUPERVISED boundary signal.

The measurements say what is missing. Reconstruction alone learns texture -- the same
29,128-parameter architecture trained that way scored 0.2193 against our 224-parameter
baseline's 0.2658, calling 43.5% of the image foreground against a true 8.8%. Mask
supervision learns boundaries and reaches 0.4252. The difference is whether the objective
knows that features should be flat inside an object and jump at its edge.

Total variation says exactly that, with no labels. Penalising the L1 norm of the spatial
gradient of the feature map pushes it piecewise constant: flat within a region, free to
jump across one. It targets the quantity measured as our worst -- within-object gamma
dissimilarity 0.0712 on Multi-dSprites against 0.0523 on CLEVR and 0.0086 on Tetrominoes,
which orders exactly like fragmentation.

Reconstruction alone cannot flatten, because it must reproduce texture. The two terms pull
against each other, and the weight sets where they balance.

Diagnostic B showed their encoder, swapped into our core and readout, lifts Multi-dSprites
FG-ARI 0.2658 -> 0.4574 and matched-object IoU 0.5442 -> 0.7601 on 40 images, with groups
per image 63.7 -> 41.6. That is about half the distance to their 0.6833.

But their weights are mask-supervised (config mask_supervised = True), so the number is
supervised-source and cannot go in a table beside CAE, CAE++, CtCAE, Rotating Features or
SynCx. What we can take is the architecture: 224 base parameters frozen, plus a dilated
2/4/8 context residual with a zero-initialised output, 29,128 in total against our 224.

The open question this answers: can a reconstruction objective learn what mask supervision
taught that architecture? The zero-init means training starts exactly at our current
encoder, so any gain is attributable to the context.

TRAIN frames only, ids [0, 6000). No mask is read anywhere.
"""
import argparse, json, sys, time
import numpy as np, torch
from torch import nn
sys.path.insert(0, "/export_home/tkim1")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")
sys.path.insert(0, "/export_home/tkim1/collaborative_test/ported")
from input_layer_generator import CNNFeatureEncoder, CNNFeatureDecoder
from context import ContextualFeatureEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True, help="uint8 [N,128,128,3] TRAIN frames")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--tv", type=float, default=0.0, help="total-variation weight on the feature map")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

rgb = torch.from_numpy(np.load(a.images))
assert rgb.dtype == torch.uint8 and rgb.shape[1:] == (128, 128, 3), rgb.shape
print("TRAIN frames %s" % (tuple(rgb.shape),), flush=True)

enc = ContextualFeatureEncoder(CNNFeatureEncoder(8, 3, 3, bias=True)).to(a.device)
dec = CNNFeatureDecoder(8, 3, out_channels=3, bias=True).to(a.device)
# the base encoder is frozen inside ContextualFeatureEncoder, so only the context and the
# decoder learn. That is the point: the base is already our 224-parameter encoder.
trainable = [p for p in enc.parameters() if p.requires_grad] + list(dec.parameters())
print("context parameters %d, decoder %d, frozen base %d"
      % (sum(p.numel() for p in enc.context.parameters()),
         sum(p.numel() for p in dec.parameters()),
         sum(p.numel() for p in enc.base_encoder.parameters())), flush=True)
opt = torch.optim.Adam(trainable, lr=a.lr)
n = rgb.shape[0]
first = last = None
for ep in range(a.epochs):
    perm = torch.randperm(n)
    tot = cnt = rt = tt = 0
    t0 = time.time()
    for s in range(0, n, a.batch_size):
        x = rgb[perm[s:s + a.batch_size]].to(a.device).permute(0, 3, 1, 2).float() / 255.0
        f = enc(x)
        y = dec(f)
        recon = nn.functional.mse_loss(y, x)
        # L1 of the spatial gradient, per feature channel. Flat inside a region, free to
        # jump across an edge -- the prior reconstruction has no way to express.
        tv = ((f[:, :, 1:, :] - f[:, :, :-1, :]).abs().mean()
              + (f[:, :, :, 1:] - f[:, :, :, :-1]).abs().mean())
        loss = recon + a.tv * tv
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tot += float(loss.detach()) * x.size(0); cnt += x.size(0)
        rt += float(recon.detach()) * x.size(0); tt += float(tv.detach()) * x.size(0)
    m = tot / cnt
    if first is None: first = m
    last = m
    print("  epoch %2d/%d  total %.6f  recon %.6f  tv %.6f  (%.0fs)"
          % (ep + 1, a.epochs, m, rt / cnt, tt / cnt, time.time() - t0), flush=True)
print("reconstruction %.6f -> %.6f" % (first, last))
# the gamma builder needs the feature statistics measured on the same TRAIN frames
tot = sq = cv = 0
enc.eval()
with torch.no_grad():
    for s in range(0, n, 50):
        f = enc(rgb[s:s + 50].to(a.device).permute(0, 3, 1, 2).float() / 255.0).double()
        tot += f.sum().item(); sq += (f * f).sum().item(); cv += f.numel()
mu = tot / cv; sd = (sq / cv - mu * mu) ** 0.5
print("TRAIN feature statistics: mean %.6f  std %.6f" % (mu, sd))
torch.save({"encoder_state": enc.state_dict(),
            "feature_mean": torch.tensor(float(mu)), "feature_std": torch.tensor(float(sd)),
            "config": {"feature_clip": 3.0, "mask_supervised": False},
            "tv_weight": a.tv, "epochs": a.epochs,
            "supervision": "reconstruction + total variation, no masks"}, a.out)
print("saved", a.out)
