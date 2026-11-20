"""Train the peer's contextual encoder architecture by reconstruction -- unsupervised.

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
    tot = cnt = 0
    t0 = time.time()
    for s in range(0, n, a.batch_size):
        x = rgb[perm[s:s + a.batch_size]].to(a.device).permute(0, 3, 1, 2).float() / 255.0
        y = dec(enc(x))
        loss = nn.functional.mse_loss(y, x)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tot += float(loss) * x.size(0); cnt += x.size(0)
    m = tot / cnt
    if first is None: first = m
    last = m
    print("  epoch %2d/%d  recon %.6f  (%.0fs)" % (ep + 1, a.epochs, m, time.time() - t0), flush=True)
print("reconstruction %.6f -> %.6f" % (first, last))
torch.save({"encoder_state": enc.state_dict(),
            "context_params": sum(p.numel() for p in enc.context.parameters()),
            "epochs": a.epochs, "supervision": "reconstruction only, no masks"}, a.out)
print("saved", a.out)
