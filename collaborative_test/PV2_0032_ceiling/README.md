# PV2_0032 — fg_ari 0.75 is not reachable by readout work

**completed. ORACLE diagnostics, reported separately, never scores.** Parent
`PV2_0031`. Validation 300, seeds 0/1/2, window 1024, kernel sigma 1.0.

## Why measure this

Every tuning axis was exhausted with the score at 0.7059 and the goal at 0.75, so
the question became whether 0.75 is attainable at all rather than which knob to
turn next.

`fg_ari` is computed over foreground patches only, so an oracle **foreground mask**
cannot move it -- the peer measured exactly that, 0.357 unchanged. The cluster
**count** is the one oracle that can, so spectral clustering was given the true
object count per image.

## Result

| | fg_ari (3-seed) |
| --- | --- |
| spatial kernel alone + true count -- the floor | 0.4127 |
| **spike affinity + true count** | **0.6360** |
| spike affinity, best rule (connected components) | **0.7059** |
| phase affinity, best rule | 0.7255 |
| **phase affinity + true count -- the ceiling** | **0.7338** |
| the goal | **0.75** |

## Two conclusions

**Count inference is not the bottleneck.** Handed the true count, the spike
readout scores 0.6360 -- *below* the 0.7059 it reaches with connected components
and no oracle at all. The under-counting noted earlier (5.89 predicted against
6.20 true) is real but costs nothing in fg_ari. What limits fg_ari is the affinity
itself.

**0.75 is above this model's ceiling for any readout.** Using the true object
count *and* the phase readout, which is a diagnostic and cannot be a score,
fg_ari reaches 0.7338 -- 0.016 short of the goal. No grouping rule, threshold,
window or kernel can close that, because the best of them is already measured
above and still falls short with an oracle attached.

The remaining gap is in the synchrony structure the model produces, not in how it
is read. That needs a core that binds better, not a better readout.

## Which points at the one untouched lever

From a comment in `train_s2net_core.py`, on the feature encoder:

> trained as a reconstruction autoencoder and never asked to separate objects;
> per-channel IoU against ground truth averages 0.079, with two channels at
> exactly 0.000, so the features cap everything downstream.

Kuramoto can only bind what the encoder has distinguished. Every experiment on
this branch has moved the oscillators, the graph, the spiking layers, the
objective or the readout; the features feeding all of it were never trained for
this task. `--encoder-lr` already exists, and training the encoder alongside the
core leaves SNN + Kuramoto bidirectional intact, so it is within the research
question.

That is `PV2_0033`.
