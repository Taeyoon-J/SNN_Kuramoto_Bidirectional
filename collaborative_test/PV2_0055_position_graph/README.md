# PV2_0055 — position in the graph makes the loss easier and the score worse

**completed. Negative result with a mechanism, and it closes the last open route
to condition 3.** Parent `PV2_0053`, correcting `PV2_0054`. Validation 300,
contract v1, seed 0, scored from spike-classifier masks.

## The hypothesis

`PV2_0053`: `graph_generator.py:123` builds every edge from the cosine of eight
appearance channels, and position never enters that code — it arrives only as a
distance prior on the logits and as the readout's Gaussian kernel. 38.8% of
validation images hold an object pair whose centroids sit closer together than the
objects are internally spread, pairs that cosine cannot separate. Appending
coordinates takes that to 28.4% at w=0.5, 18.1% at w=1 and 7.7% at w=2.

## The code change

`PV2_0054` widened gamma and the spike path died, because `_to_phase` standardises
jointly over channels so two extra channels rescaled the appearance signal. This
adds `graph_extra_channels = n`: the last n channels reach the coupling graph and
are stripped before anything else sees them.

```
graph_generator  <- all 10 channels     (appearance + position)
phase path       <- first 8 channels    (appearance only, unchanged)
```

Kuramoto, the gate, the dendrite, the membrane and the spike readout are untouched,
so the per-image standardisation and the absolute threshold are exactly as before.
Files: `hyperparameter.py` (field and range check), `s2net_cls.py` (split, and the
feedback graph call keeps the full-width tensor), `training/train_s2net_core.py`
(`--graph-extra-channels`), `collaborative_test/evaluate_model.py` (both widths read
from the checkpoint).

**Regression gate.** The first row of every run is the 8-channel path on an existing
checkpoint, and nothing below it is read unless that reproduces its recorded value.
It earned its place immediately: the first version of the patch inserted the strip
between the `elif` and `else` of the graph-selection chain, which rebound that
`else` to the new `if`, so at `graph_extra_channels = 0` the learned graph `sc` was
overwritten by the static one. The 8-channel path scored **0.0033** instead of
0.7239 — a bug that fired only in the configuration every existing result uses,
while the 10-channel smoke test passed. Fixed by placing the strip after the whole
chain; three checkpoints then reproduced 0.7239 / 0.7251 / 0.4951,
0.7235 / 0.7164 / 0.4935 and 0.7275 / 0.7148 / 0.4974 to four decimals.

## Result

| weight | sync | fg_ari | fg_iou | obj_iou | groups | d fg_ari |
| --- | --- | --- | --- | --- | --- | --- |
| **bar, 8ch** | 0.05 | **0.7239** | **0.7251** | **0.4951** | 5.46 | — |
| 0.5 | 0.20 | 0.4703 | 0.5678 | 0.2495 | 2.44 | **-0.2536** |
| 1 | 0.20 | **0.5556** | 0.6443 | 0.4221 | 5.91 | **-0.1683** |
| 2 | 0.20 | 0.4575 | 0.2947 | 0.3330 | 9.05 | **-0.2664** |

**Position in the graph costs 0.168 at best.** The hypothesis is refuted.

One thing did reproduce: `PV2_0053` predicted w=1 as the peak of the scale-invariant
trade, min-pair over within-object spread (1.530, 1.588, **1.645**, 1.630, 1.463),
and w=1 is indeed the best of the three. The ordering transferred; the level did not.

## Why it hurts — the loss got easier

The trained losses are **lower** than the baseline's: 21.86 -> **13.91** at w=0.5,
22.49 -> 17.31 at w=1, 22.34 -> **13.75** at w=2, against 32.49 -> 17.86 for the
8-channel run. The model satisfied its objective *better* while scoring *worse*.

Position gives the graph a shortcut. The PLV terms — bimodality, balance, target
density — ask for groups that are internally coherent and roughly the right size,
and **spatially compact blobs satisfy that without tracking objects at all.**
`PV2_0043` already measured what geometry alone is worth under this readout:
**0.3902**. PG_w2 lands at 0.4575 with 9.05 groups and 0.31 predicted foreground —
near that geometry-only level. PG_w0.5 collapses the other way, to 2.44 groups
against the baseline's 5.46.

So the objective is partly satisfiable by geometry, and feeding geometry into the
code it is computed from makes the shortcut available. This is the failure the peer
reported for their readout and which `PV2_0043` had ruled out for *ours* — the
spatial kernel there acts only at readout, where spikes still contribute +0.217.
Putting geometry inside the trained graph is a different thing.

## What this means for the goal

Condition 1 stands unmet at **0.7250** against 0.75. Eight feature routes are now
closed, and this one adds a reason the others may share: **the training objective,
not only the encoder, limits the features.** `PV2_0034` reached 0.9754 by handing
oracle features to the *existing, unretrained* core, so the readout can exploit good
features — but no objective tried so far creates them, and this one shows the
objective can be satisfied without them.

A route to 0.9 therefore needs a training signal that cannot be satisfied by
geometry. Reconstruction is the only such signal tested (`PV2_0039`, 0.6779). The
peer is running cross-view spike consistency (their SW0139), which is another, and
worth reviewing before duplicating.
