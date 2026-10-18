# PV2_0097 — a term for the number of groups

**Status:** completed, seed 0. **Parent:** `PV2_0096`. **Contract:** v2.
**Hypothesis:** the 32x32 failure is a group count — 19.5 against 6.49 true objects — and
three readout knobs could not move it, so the control belongs in the objective.
`loss_function.py:598` already holds `plv_group_count_loss`, which targets the effective
group count through the participation ratio `N^2/||PLV||_F^2`, and the published
configuration leaves its weight at zero.
**Gate, registered before the run:** the gain must appear at predicted foreground <= 0.20;
it must not be bought with foreground IoU; and `patch_matched_object_iou` must move too.

## Result, 8 epochs, seed 0, validation 300 (from `PV2_0101`'s matched re-scoring)

| weight | fg_ari | fg_iou | obj_iou | groups | pred_fg |
| --- | --- | --- | --- | --- | --- |
| 0 (control) | 0.7699 | 0.7531 | 0.7017 | 15.84 | 0.1512 |
| **5** | **0.7803** | **0.7807** | 0.6914 | **13.68** | 0.1444 |
| 20 (validation 200) | 0.8095 | 0.3485 | 0.6679 | 10.97 | **0.4065** |

Against the gate, at weight 5:

| condition | result |
| --- | --- |
| 1. gain at predicted foreground <= 0.20 | **pass** — 0.1444 |
| 2. not bought with foreground IoU | **pass** — fg_iou *rises* +0.0276 |
| 3. `matched_object_iou` must move | **fail** — -0.0103 |

**Two of three.** The term does what it says — groups fall 15.84 -> 13.68 toward the true
6.49, and fg_ari and foreground IoU both improve — but per-object IoU slips slightly, so
this is not the clean pass the gate was written to require.

Weight 20 is the gate's whole purpose: **fg_ari +0.0361 while foreground IoU falls 0.4086
and predicted foreground reaches 0.4065.** Read on fg_ari alone it is the best arm. It is
the fourth time in this project that reading fg_ari alone would have produced a wrong
verdict.

## A numerical problem worth recording

The term is wildly out of scale at initialisation. First-epoch loss is **4.89e24** at
weight 5 and **1.95e25** at weight 20, against the control's 62.77. The participation
ratio is `N^2 / ||PLV||_F^2`, and an initial PLV matrix near zero sends it to ~1e13.

Both arms recover — weight 5 ends at 42.46 with `plv_group_count` 0.1527, weight 20 at
2198.32 with 21.29 — and the collapse trajectories stay healthy (6.40 -> 2.54 and
7.37 -> 3.72 against the control's 4.72 -> 2.58). But at weight 20 the collapse is visibly
worse than the control's, so **the term's effect and numerical instability are mixed
together at high weight**, and that arm's numbers should not be read as the term's
behaviour. A warmup or a clamp on the ratio would separate them.

## What the experiment actually produced

Its control, not its treatment. See `PV2_0101`: at 8 epochs the control alone beats the
published 16x16 configuration on all three metrics, and `PV2_0094`'s 40-epoch budget was
the error. The group-count term adds a further +0.0104 fg_ari and +0.0276 foreground IoU
on top of that, for -0.0103 obj_iou.

## Next

`PV2_0102` runs seeds 1 and 2 of both the control and weight 5. If the term holds its two
of three across seeds it is worth a weight sweep between 5 and 20 with the scale problem
fixed; if it does not, the control alone is the result.
