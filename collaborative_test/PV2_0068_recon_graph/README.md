# PV2_0068 / PV2_0069 — reconstruction does not move the graph, and the graph does not amplify its own groups

**completed. Fifteenth and sixteenth routes closed.** Parent `PV2_0066`. Validation
300, contract v1, seed 0 (0068) and three graph-frozen checkpoints (0069).

## Why these two, and not more feature transforms

`PV2_0066` showed the graph carries 94% of the oracle's gain and does it in the
calibrated regime (0.9671 / 0.9353 / 0.7584 at predicted foreground 0.1121). Fourteen
routes had failed to improve the graph's **input**. These are the two ways left that do
not require object labels: train the graph under a loss that knows what an object is,
and feed the graph the model's own object estimate.

## `PV2_0068` — reconstruction with the features frozen

`slot_reconstruction_loss` compares the phase grouping against mean RGB per patch, a
target external to the model that cannot be gamed by changing the features. It was
unreachable with frozen features because `train_s2net_core.py:201` only builds the
target under `--image-dir`, which also makes the encoder live. `PV2_0039` trained the
encoder that way (0.6779) and `PV2_0045` froze the graph and trained the rest;
**freezing the features and training the graph under reconstruction had never been
run.** `--recon-target-path` precomputes the pooled RGB so it can.

Verified before the real runs: one epoch gives loss 24.338 against the control's
23.524 with `slot_reconstruction = 0.5258` live and nonzero, and the targets load
aligned (1,000 rows for 1,000 training rows).

| arm | fg_ari | fg_iou | obj_iou | pred_fg | final loss |
| --- | --- | --- | --- | --- | --- |
| **bar, no term** | **0.7239** | **0.7251** | 0.4951 | 0.1290 | 32.49 -> **17.86** |
| weight 0.5 | 0.4141 | 0.4533 | 0.3234 | 0.1525 | — |
| weight 1.0 | 0.6392 | 0.4905 | 0.4678 | 0.2198 | 24.34 -> 18.06 |
| weight 2.0 | 0.6822 | 0.4586 | 0.4904 | 0.2503 | 24.60 -> 18.70 |
| weight 1.0, from the trained core | **0.7175** | 0.7035 | 0.4958 | **0.1289** | 18.37 -> 15.49 |
| oracle features to the graph (diagnostic) | 0.9671 | 0.9353 | 0.7584 | 0.1121 | — |

The optimisation worked — `slot_reconstruction` settles near 0.70 and the loss reaches
18.06 against the reference 17.86 — but the best calibrated result is **0.7175,
-0.0064 on the bar**. Reconstruction trains the graph to explain the image and that
does not make it group objects better.

## `PV2_0069` — the graph fed its own groups as one-hot codes

Oracle codes are one-hot per object, so between objects they are orthogonal by
construction, which is exactly the property fifteen routes could not synthesise from
appearance. But the model already predicts instance identity at fg_ari 0.7239. One-hot
codes built from **its own predictions** are a noisy oracle input, and the question is
whether the graph amplifies them.

This is not `PV2_0037`'s self-bootstrap, which fed the groups back as features for the
whole pipeline (+0.006). Here they go to the graph alone with the phase path on real
gamma — the split `PV2_0066` showed matters — and the comparison is against the
accuracy of the input groups, so amplification is measured rather than assumed.

| | fg_ari | fg_iou | obj_iou | pred_fg |
| --- | --- | --- | --- | --- |
| input groups (what the codes were built from) | **0.7239** | 0.7251 | 0.4951 | 0.1290 |
| codes -> graph, FZG_s0 | 0.7214 | 0.7212 | 0.4830 | 0.1271 |
| codes -> graph, BIM6_s1 | 0.7174 | 0.7183 | 0.4767 | 0.1250 |
| codes -> graph, FZG_s2 | 0.7210 | 0.7209 | 0.4823 | 0.1271 |
| **mean** | **0.7199** | 0.7201 | 0.4807 | 0.1264 |
| oracle one-hot -> graph | **0.9671** | 0.9353 | 0.7584 | 0.1121 |

**0.7199 out for 0.7239 in: a fixed point, very slightly contractive downward.** The
graph does not amplify the model's own estimate, so iterating cannot climb. Oracle
one-hot gives 0.9671 and 0.72-accurate one-hot gives 0.72, which says the errors in the
predicted groups are precisely the errors that survive the graph.

64 of 300 images had more than 8 groups and were truncated to the core's channel width,
which is a limitation of the test rather than of the idea; but a fixed point that sits
0.004 **below** its input is not a near-miss.

## Where this leaves part 3 of the goal

**Feasibility is established and the method is not.** The architecture reaches
0.9671 / 0.9353 / 0.7584 in the calibrated regime, unmodified, when the graph's input
separates objects. Sixteen routes have failed to produce that input:

- eleven on feature geometry, three of which improved a separation statistic without
  moving fg_ari (`PV2_0058`, `PV2_0062`, `PV2_0064`), which is why separation statistics
  are not sufficient statistics for the score;
- two on the graph's input confined to the graph (`PV2_0067`), failing in opposite
  directions — margin without calibration, calibration without margin;
- one on a representation trained by reconstruction with slot competition on our own
  data (`PV2_0065`, the matched Slot encoder: within-object cosine 0.0019, near the
  oracle's zero, but between-object 0.0316, eight times too small);
- one on training the graph under reconstruction (`PV2_0068`);
- one on the graph amplifying its own groups (`PV2_0069`).

What the oracle supplies is **instance identity**, and every label-free route tried
either reproduces the geometry without the information or is a fixed point at the
accuracy it starts from. The remaining untested source is cross-view consistency as a
training objective, which the peer has queued as SW0139 and which had no result at
three fetches this session.
