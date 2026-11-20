"""What AUC does a given gamma-contrast correlation imply?

PV2_0115's gate is an AUC on real checkpoints and needs the server. This is the
arithmetic that lets the training log be read before then. The loss is the negative
per-image Pearson correlation between mean spike activity and the gamma deviation,
and GCON_w1's log shows gamma_contrast = -0.3102, i.e. correlation 0.3102.

The deviation separates true foreground at AUC 0.947 (measured, 200 images,
label-free). If activity tracked it perfectly, activity would inherit that 0.947.
At correlation r it inherits less. This maps r to the AUC it buys, under a normal
model fitted to the measured foreground rate and the measured 0.947.

Offline, no server, no checkpoints. It predicts the gate rather than deciding it:
the real AUC comes from auc_check.py on the trained arms.
"""
import numpy as np
from math import sqrt, erf

N, FG = 1024, 0.119          # patches per image, measured foreground fraction
TARGET_AUC = 0.947           # measured AUC of the gamma deviation
rng = np.random.default_rng(0)

# Pick the separation that reproduces AUC 0.947 for two unit-variance normals:
# AUC = Phi(mu / sqrt(2))
from scipy.stats import norm
mu = norm.ppf(TARGET_AUC) * sqrt(2.0)
print("deviation model: foreground mean %.3f, background 0, unit variance" % mu)

def auc(score, pos):
    r = np.argsort(np.argsort(score))
    npos = int(pos.sum()); nneg = len(pos) - npos
    return (r[pos].sum() - npos * (npos - 1) / 2) / (npos * nneg)

IM = 400
y = rng.random((IM, N)) < FG
d = rng.standard_normal((IM, N)) + mu * y          # the gamma deviation
check = auc(d.ravel(), y.ravel())
print("model reproduces deviation AUC %.3f (target %.3f)" % (check, TARGET_AUC))
print()
print("  correlation r    activity AUC     note")
noise = rng.standard_normal((IM, N))
dz = (d - d.mean(1, keepdims=True)) / d.std(1, keepdims=True)
nz = (noise - noise.mean(1, keepdims=True)) / noise.std(1, keepdims=True)
for r in (0.0, 0.10, 0.20, 0.3102, 0.40, 0.50, 0.70, 0.90, 1.0):
    a = r * dz + sqrt(max(0.0, 1 - r * r)) * nz
    v = auc(a.ravel(), y.ravel())
    note = ""
    if abs(r - 0.3102) < 1e-6: note = "<-- GCON_w1's logged value"
    if r == 0.0: note = "chance, where the spikes are now (0.494 measured)"
    if r == 1.0: note = "the ceiling this term can buy"
    print("   %5.3f           %.3f            %s" % (r, v, note))
print()
print("Reading: the gate asks the activity AUC to rise from 0.494. Under this model a")
print("correlation of 0.31 lands near the value printed above -- real signal, well short")
print("of the deviation's own 0.947. Whether that is enough for the three metrics is")
print("exactly what the falsifier was registered to decide, and it is not decided here.")
