# PV2_0077 — training under the superlinear cost is harmful; the gain is inference-only

**completed. Twenty-third route closed, and it bounds what `PV2_0076` is.** Parent
`PV2_0076`. Seed 0, validation 300, contract v1.

## Why it had to be run

`PV2_0076`'s superlinear step cost took test from 0.7533 to 0.7603 with all three metrics
up, but two things in it were not learned: the contrast of 8 was **written** into the
checkpoint, and every other weight was fitted under the **linear** cost. `PV2_0075`'s
arithmetic predicted that training should prefer a different contrast under a squared
cost, since a boundary crossing is then charged 27.7x a shading step rather than 5.3x.

## Result

| setting | fg_ari | fg_iou | obj_iou | pred_fg |
| --- | --- | --- | --- | --- |
| trained p1, eval p1 (bar) | **0.7239** | **0.7251** | 0.4951 | 0.1290 |
| trained p1, eval p2 + contrast 8 (`PV2_0076`) | **0.7327** | 0.7255 | 0.5033 | 0.1291 |
| trained p2, eval p2 | 0.4268 | 0.4115 | 0.3035 | 0.1371 |
| trained p2, eval p1 (control) | 0.4249 | 0.4136 | 0.3052 | 0.1405 |
| trained p3, eval p3 | 0.5248 | 0.4353 | 0.3493 | 0.1804 |
| trained p3, eval p1 (control) | 0.5254 | 0.4221 | 0.3584 | 0.1883 |

**Training under the cost is much worse**, and the control rows say where the damage is:
evaluating the same checkpoints at power 1 gives 0.4249 and 0.5254, essentially
unchanged. **The weights are bad, not the evaluation.**

Two further signals point the same way. The **loss went down**: 16.71 and 16.80 against
the power-1 reference 17.86, so the objective was satisfied *better* while the score
collapsed — the pattern `PV2_0055` recorded for position in the graph. And training
**lowered** the contrast, 2.1732 -> 1.9588, so under a sharper cost it chose to rely on
the geodesic selectivity *less*, not more. The prediction that it would prefer a stronger
contrast was wrong.

## What this establishes about `PV2_0076`

Its gain is specifically an **inference-time correction**: a sharper cost applied to
weights fitted under a softer one. The sharper cost is a better *readout* of the graph
those weights produce, and a worse *objective* to fit them with — presumably because a
sparser, sharper graph during training gives the PLV terms an easier, more degenerate way
to be satisfied, which is the failure mode this project has now seen four times
(`PV2_0055`, `PV2_0058`, chromaticity, and here).

So `PV2_0076` stands as the session's best result and it stands as what it is: a
post-hoc correction worth +0.0070 on test, not a better way to train.

## Where part 3 stands

Unmet. 0.7603 on test against a 0.9 target. `PV2_0075`'s remaining candidates — a fixed
per-step toll so a long cheap corridor costs something, and making the cost's *shape*
learnable rather than its scale — are untested, but this result tempers the expectation:
the one change of that family tried so far helps only when kept out of the objective.
