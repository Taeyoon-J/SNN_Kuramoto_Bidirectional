"""The same context encoder, trained with MASKS instead of reconstruction.

This answers one question directly: if mask supervision is what makes the peer's encoder
worth +0.19 on Multi-dSprites, does OUR pipeline gain the same when we supervise our own?

Reconstruction already failed here -- the identical architecture trained to reproduce
pixels scored 0.2193 against our 224-parameter baseline's 0.2658, calling 43.5% of the
image foreground against a true 8.8%. It learned texture, and the dilated receptive field
spread it further.

The objective here is the simplest thing masks can say: patches of the same object should
have similar features, patches of different objects should not. Per image, over the patch
grid, pull within-object feature pairs together and push across-object pairs apart. This is
not the peer's exact teacher -- theirs builds within-object frequency targets with a gamma
anchor -- but it is the same information source, and the question is whether the
information helps at all, not whether we reproduce their recipe.

TRAIN ids [0, 6000) only. This run is SUPERVISED by construction and anything it produces
is barred from the unsupervised table; it exists to size the effect.
"""
import argparse, json, sys, time
import numpy as np, torch
from torch import nn
sys.path.insert(0, "/export_home/tkim1")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")
sys.path.insert(0, "/export_home/tkim1/collaborative_test/ported")
from input_layer_generator import CNNFeatureEncoder
from gamma_initializer import FeaturePatchGammaInitializer
from context import ContextualFeatureEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--targets", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--grid", type=int, default=64)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3)
a = ap.parse_args()

rgb = torch.from_numpy(np.load(a.images))
lab = torch.load(a.targets, map_location="cpu")["patch_labels"][:rgb.shape[0]].long()
assert lab.shape[1] == a.grid, (lab.shape, a.grid)
print("TRAIN %s, patch labels %s" % (tuple(rgb.shape), tuple(lab.shape)), flush=True)

enc = ContextualFeatureEncoder(CNNFeatureEncoder(8, 3, 3, bias=True)).cuda()
patch = FeaturePatchGammaInitializer(grid_size=a.grid)
opt = torch.optim.Adam([p for p in enc.parameters() if p.requires_grad], lr=a.lr)
print("trainable context parameters %d"
      % sum(p.numel() for p in enc.context.parameters()), flush=True)

n = rgb.shape[0]
first = last = None
for ep in range(a.epochs):
    perm = torch.randperm(n); tot = cnt = 0; t0 = time.time()
    for s in range(0, n, a.batch_size):
        idx = perm[s:s + a.batch_size]
        x = rgb[idx].cuda().permute(0, 3, 1, 2).float() / 255.0
        y = lab[idx].cuda().reshape(len(idx), -1)              # [B, N]
        g = patch(enc(x))                                      # [B, 8, N]
        g = nn.functional.normalize(g, dim=1)
        sim = torch.einsum("bcn,bcm->bnm", g, g)               # cosine, [B, N, N]
        same = (y.unsqueeze(2) == y.unsqueeze(1)).float()
        fg = (y > 0).float()
        pair_fg = fg.unsqueeze(2) * fg.unsqueeze(1)            # foreground pairs only
        eye = torch.eye(y.size(1), device=y.device).unsqueeze(0)
        m = pair_fg * (1.0 - eye)
        pos = (m * same * (1.0 - sim)).sum() / m.mul(same).sum().clamp_min(1.0)
        neg = (m * (1.0 - same) * sim.clamp_min(0.0)).sum() / m.mul(1.0 - same).sum().clamp_min(1.0)
        loss = pos + neg
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tot += float(loss.detach()) * len(idx); cnt += len(idx)
    m_ = tot / cnt
    if first is None: first = m_
    last = m_
    print("  epoch %2d/%d  pull+push %.6f  (%.0fs)" % (ep + 1, a.epochs, m_, time.time() - t0), flush=True)
print("objective %.6f -> %.6f" % (first, last))

# feature statistics on the same TRAIN frames, as the gamma builder needs
tot = sq = cntv = 0
enc.eval()
with torch.no_grad():
    for s in range(0, n, 50):
        f = enc(rgb[s:s + 50].cuda().permute(0, 3, 1, 2).float() / 255.0).double()
        tot += f.sum().item(); sq += (f * f).sum().item(); cntv += f.numel()
mu = tot / cntv; sd = (sq / cntv - mu * mu) ** 0.5
print("TRAIN feature statistics: mean %.6f  std %.6f" % (mu, sd))
torch.save({"encoder_state": enc.state_dict(),
            "feature_mean": torch.tensor(float(mu)), "feature_std": torch.tensor(float(sd)),
            "config": {"feature_clip": 3.0, "mask_supervised": True}}, a.out)
print("saved", a.out)
