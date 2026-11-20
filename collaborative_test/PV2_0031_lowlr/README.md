# PV2_0031 — the peer's low-LR rescue does not transfer

**failed to transfer.** Parent `PV2_0027`. Source: `origin/patch_v2_sw` SW0050 and
SW0052. Validation 300, seeds 0/1/2 and the broken 3/4/5, readout window 1024,
kernel sigma 1.0.

## Why it was tried

The peer's seed 2 collapsed and a lower learning rate with early stopping
rescued it: FG-ARI 0.186 at lr 1e-3 / 40 epochs, 0.347 at lr 3e-4 / 40 epochs,
and 0.396 at lr 3e-4 / epoch 25, which they selected. **lr 1e-3 for 40 epochs is
exactly this branch's setting** -- the one that collapsed for them -- and epoch 25
beating epoch 40 is an over-training signal worth having.

## Result

| | fg_ari (3-seed) | per seed | range |
| --- | --- | --- | --- |
| lr 1e-3, 40 epochs | **0.7059** | 0.664 / 0.724 / 0.730 | 0.066 |
| lr 3e-4, 25 epochs | 0.6657 | 0.679 / 0.644 / 0.674 | **0.035** |

The mean falls 0.040. Spread halves, which is the effect they were after, but
the level is given up to get it. Foreground IoU is worse too: `LLR_s0` drops to
0.3774 with predicted foreground 0.348.

And it does not repair the broken seeds. Seed 3 stays at foreground IoU 0.185
with predicted foreground 0.391, seed 4 at 0.230 and 0.452, and **seed 5 is still
completely dead** -- 0.0033, firing rate 0.

## Which supports the diagnosis in PV2_0023

Seed 5 survives a lower learning rate unchanged and dead, while the
inference-side threshold fix revives it from foreground IoU 0.0000 to 0.6353
with no weight touched. So its failure is not an optimisation failure; it is the
membrane threshold sitting outside the membrane's range.

The peer's collapse has a different shape: their seed 2 shows a phase product-PLV
mean of 0.4454 against 0.715 and 0.649 for their other seeds, a phase-side
collapse. Here the phase side is healthy and the threshold is misplaced. Same
symptom, different cause, so the remedy does not carry.

## A structural note on the 0.75 target

| readout | fg_ari |
| --- | --- |
| spike -- the score | 0.7059 |
| phase -- DIAGNOSTIC | 0.7255 |
| the goal | 0.75 |

The goal sits **0.024 above the phase readout**. Reaching it therefore requires
the spike readout not to match theta's own readout but to beat it, and every
measurement so far points the other way: the gap is 0.0196 under connected
components and 0.017 under spectral clustering, and it closed only with readout
window length, which saturates.

That does not make 0.75 impossible -- the two readouts are different measurements
of the same dynamics, not nested bounds -- but it does mean no remaining tuning
axis can plausibly get there, since all of them are exhausted:

| axis | state |
| --- | --- |
| coupling graph | spent, +0.06 even with ground truth |
| classifier knobs | at best |
| synchrony threshold | flat 0.05-0.20 |
| readout window | saturates at 1024, worth +0.18 from 64 |
| training window | refuted |
| spike synchrony weight | 5 is the peak |
| bimodality weight | 6 is the peak at two windows |
| phase-to-spike imitation | five attempts failed |
| spectral readout | fg_ari inferior, obj_iou better |
| learning rate and epochs | this experiment, worse |
