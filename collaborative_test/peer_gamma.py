"""Diagnostic B: does the peer's encoder fix our affinity?

Our Multi-dSprites failure is fragmentation -- 4.24 predicted groups per true object at
grid 32, 65.4 groups per image at grid 64 for 2.37 objects -- and eight candidates are now
eliminated by measurement (PV2_0128): the grid, the readout settings across twelve
combinations, sprite overlap, object/background contrast, within-image uniformity, the
dataset variant, replicating the peer's own readout settings, and porting their merge stage.

What is left is the thing that makes Q. This swaps in the peer's encoder -- weights and all
-- and measures the affinity our own core then produces.

This is a DIAGNOSTIC, not a result. Their encoder is mask-supervised (config
mask_supervised = True), so any number from it is supervised-source and never enters an
unsupervised table. The question it answers is only: is the encoder where our fragmentation
comes from?
"""
import sys, json, argparse
import numpy as np, torch
sys.path.insert(0, "/export_home/tkim1/collaborative_test")
sys.path.insert(0, "/export_home/tkim1/collaborative_test/ported")
sys.path.insert(0, "/export_home/tkim1")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")

from input_layer_generator import CNNFeatureEncoder
from gamma_initializer import FeaturePatchGammaInitializer
from context import ContextualFeatureEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--images", default="/work/USERS/tkim1/mod/images128_multi_dsprites_validation.npy")
ap.add_argument("--out", required=True)
ap.add_argument("--grid", type=int, default=64)
a = ap.parse_args()

blob = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
enc = ContextualFeatureEncoder(CNNFeatureEncoder(8, 3, 3, bias=True))
missing, unexpected = enc.load_state_dict(blob["encoder_state"], strict=False)
assert not missing and not unexpected, ("encoder state does not match", missing, unexpected)
enc.eval().cuda()
patch = FeaturePatchGammaInitializer(grid_size=a.grid)
mean = blob["feature_mean"]; std = blob["feature_std"]; clip = float(blob["config"]["feature_clip"])
print("loaded %s  mask_supervised=%s  mean %.6f  std %.6f  clip %.1f"
      % (a.checkpoint.split("/")[-1], blob["config"]["mask_supervised"],
         float(mean), float(std), clip))

rgb = torch.from_numpy(np.load(a.images))
assert rgb.dtype == torch.uint8 and rgb.shape[1:] == (128, 128, 3), rgb.shape
print("images %s" % (tuple(rgb.shape),))

out = []
with torch.no_grad():
    for s in range(0, rgb.shape[0], 25):
        x = rgb[s:s + 25].cuda().permute(0, 3, 1, 2).float() / 255.0
        f = enc(x)
        g = patch(((f - mean.cuda()) / std.cuda()).clamp(-clip, clip))
        out.append(g.cpu())
gamma = torch.cat(out)
assert gamma.shape[1] == 8 and gamma.shape[2] == a.grid * a.grid, gamma.shape
print("gamma %s  mean %+.4f  std %.4f" % (tuple(gamma.shape), gamma.mean(), gamma.std()))
torch.save(gamma, a.out)
print("wrote", a.out)
