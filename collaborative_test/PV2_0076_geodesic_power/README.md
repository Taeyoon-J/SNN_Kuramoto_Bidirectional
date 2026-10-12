# PV2_0076 — a superlinear geodesic step cost: all three metrics up on test

**completed. The first core change this session that improves every metric, and the
first fg_ari gain that costs no calibration.** Parent `PV2_0075`. Validation 300 for
selection, test 1000 read once for the result; three graph-frozen checkpoints.

## Derived from the measurement, not guessed

`PV2_0074` found the learned geodesic contrast already optimal — 0.5 through 32 spans
fg_ari 0.7243 down to 0.6406, with groups per image climbing 5.46 -> 6.59 as it rose, so
objects split while fg_ari fell. `PV2_0075` measured why, in the adjacent-pair
dissimilarity the geodesic actually sees:

| pair type | mean `1 - cosine` |
| --- | --- |
| within one object | 0.0331 |
| background to background | 0.0054 |
| object to background | **0.1743** |

The wall is real at **5.27x** a within-object shading step; the corridor between two
same-coloured objects is **six times cheaper** than a step inside an object. So the route
out of one object and into another is two thin walls across nearly frictionless ground,
and the contrast cannot change that because it multiplies `(1 - similarity)`
**uniformly** — walls and shading steps scale together.

What the numbers asked for was **selectivity**, and a power supplies it, because the
5.27x ratio compounds:

| power | wall / within-object |
| --- | --- |
| 1 (original) | 5.3 |
| 2 | **27.7** |
| 3 | 146 |

## The change

`graph_generator.py`, in `_geodesic_distance`:

```python
dissimilarity = (1.0 - similarity).clamp_min(0.0)
if self.geodesic_power != 1.0:
    dissimilarity = dissimilarity ** self.geodesic_power
step = euclidean * (1.0 + F.softplus(self.geodesic_contrast) * dissimilarity)
```

`geodesic_power` is a **plain float, not an `nn.Parameter`**, so unlike the contrast it
is not restored from a checkpoint and applies to the existing trained weights at
inference. **At 1.0 the expression is identical to the original**, which every run below
gates on. Wired through `hyperparameter.py`, `s2net_cls.py` and
`collaborative_test/evaluate_model.py`.

SNN and Kuramoto are untouched: the oscillators, gate, dendrite, membrane and spike
readout are exactly as they were, and only the cost function inside the graph's geodesic
relaxation changes.

## The mechanism's prediction held

At power 1.0 a contrast of 8 **hurt** (`PV2_0074`: 0.7183). Once the power makes the cost
selective, **contrast 8 is the best setting** — because the stronger factor now falls on
boundary crossings rather than on shading transitions. That is the prediction the
arithmetic made, and it is why the two combine.

## Result on test, three seeds, 1000 images

| metric | this (k=14, power 2, contrast 8) | previous (`PV2_0061`) | change | matched Slot | vs Slot |
| --- | --- | --- | --- | --- | --- |
| `patch_fg_ari` | **0.7603** +- 0.0003 | 0.7533 | **+0.0070** | 0.6663 | **+0.0940 (+14.1%)** |
| `patch_foreground_iou` | **0.3373** +- 0.0207 | 0.3358 | +0.0015 | 0.1177 | **+0.2196 (+187%)** |
| `patch_matched_object_iou` | **0.6598** +- 0.0061 | 0.6549 | +0.0049 | 0.0871 | **+0.5727 (+658%)** |

Per seed: 0.7600 / 0.7605 / 0.7604 — a spread of 0.0005.

The connected-component configuration improves too, on the same test images:
**0.7300 / 0.7132 / 0.5050** against 0.7258 / 0.7124 / 0.5029, at predicted foreground
0.1306 against a target of ~0.118.

**All three metrics up, both readouts, every seed.** On validation the same change gave
0.7633 / 0.3523 / 0.6527 against 0.7569 / 0.3449 / 0.6480, and predicted foreground
*improved* from 0.5423 to 0.5353. This is the only intervention in this session whose
fg_ari gain is not bought with foreground over-prediction — compare chroma3 (fg_ari
0.7380 at predicted foreground 0.509), spectral k=20 (0.7718 at 0.808) and whitening
(0.7670 at 0.590).

## Limitations

- **Second test read of the session.** `PV2_0061` was the first, for the same readout at
  power 1.0. The contract permits a read once a setting is settled and this one is, but
  the repetition weakens test independence and is disclosed rather than buried.
- **The gain is +0.007.** It is consistent — nine of nine validation measurements and six
  of six test ones move the same way, with fg_ari std 0.0003 — but it is not a step
  change. 0.7603 to 0.9 is still 0.14.
- The three seeds share one graph through `PV2_0044`'s freeze, so they are not three
  fully independent runs.
- `contrast = 8` is written into the checkpoint rather than trained. Training the pair
  jointly under the new cost is untested and is the obvious next step.

## What it does not settle

Part 3 of the goal asks for a route to **0.9**. This is not it. What it does establish is
that the chain **measurement -> mechanism -> change in the equations -> reproducible
gain** closes on this architecture, which twenty-one previous routes had not shown, and
`PV2_0075`'s arithmetic names the next candidates: a fixed per-step toll so a long cheap
corridor costs something, a higher power, or making the cost's shape learnable rather
than its scale.
