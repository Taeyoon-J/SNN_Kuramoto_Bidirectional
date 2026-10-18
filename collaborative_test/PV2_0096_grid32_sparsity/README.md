# PV2_0096 — graph sparsity is not the over-segmentation

**Status:** completed (negative). **Parent:** `PV2_0095`. **Contract:** v2.
**Hypothesis:** `--graph-top-k 32` is an absolute count. At 256 nodes it admits 12.5% of
the graph; at 1024 nodes **3.1%**. Four times sparser relative to the node count, so
groups may be unable to link across an object — the same class of error `PV2_0093`
confirmed for the membrane threshold and `PV2_0095` found in the spatial kernel.
**Refuted.**

## Results, validation 300, on `G32_POP40`, sigma 4.0

| top_k | fraction of nodes | d fg_ari | d fg_iou | d obj_iou | groups | pred_fg |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 3.1% | -0.0946 | -0.0160 | +0.0934 | 16.76 | 0.1276 |
| 64 | 6.3% | -0.0948 | -0.0113 | +0.0898 | 16.32 | 0.1271 |
| 128 | 12.5% (matched) | -0.0939 | -0.0073 | +0.0904 | 16.15 | 0.1265 |
| 192 | 18.8% | -0.0932 | -0.0086 | +0.0911 | 16.19 | 0.1266 |
| 256 | 25.0% | -0.0927 | -0.0084 | +0.0906 | 16.14 | 0.1266 |

Deltas are against grid 16 at matched extent (0.7239 / 0.7251 / 0.4951, pred_fg 0.1290);
true objects per image on contract v2 is 6.49.

**An eightfold denser graph moves fg_ari by +0.0019 and the group count from 16.76 to
16.14.** Nothing. The matched-fraction setting, top_k 128, is indistinguishable from the
rest.

The checkpoint was trained at top_k 32, so this arm is off-distribution and a *loss*
would have been ambiguous. It is not a loss; it is a flat line, which is unambiguous.

## What `PV2_0095` and `PV2_0096` say together

Three readout-side knobs have now been swept at grid 32 — spatial kernel width, group
size floor, graph density — and `patch_fg_ari` sits at 0.63 throughout while
`groups_per_image` stays between 16 and 20 against 6.49 true objects.

This is `PV2_0091`'s conclusion arriving independently at the other resolution: **the
readout cannot close the gap.** At grid 16 the dial maxed at 0.7718 against an external
0.8102; at grid 32 it maxes at 0.6312 against grid 16's own 0.7239. Different grids,
different error modes, same verdict about where the problem is not.

**The over-segmentation is in what the spikes encode.**

## The term the objective is missing

`loss_function.py:598` defines `plv_group_count_loss`, which targets the effective number
of synchronised groups through the participation ratio `PR = N^2 / ||PLV||_F^2` — 1 when
everything locks into one group, N when nothing locks, k for k equal blocks, and no
eigendecomposition needed. It is exposed as `--plv-group-count-weight` and
`--plv-target-groups`.

**The published configuration does not use it.** It uses `--plv-balance-weight 10.0
--plv-target-density 0.867`, and the group-count term's own docstring says it was written
to *replace* that:

> This replaces targeting a mean synchrony level, which turned out to be ill-posed: with
> CLEVR ground truth the ideal mean PLV is 0.867 if the background counts as one group
> and 0.003 if it stays incoherent, so any intermediate target pulls the solution away
> from both.

So the objective currently controls a **scalar mean synchrony** and has **no term that
controls how many groups form**, which is exactly the quantity that is wrong at grid 32 —
16 to 20 against 6.49 — and the quantity no readout knob could move.

## Next

`PV2_0097` adds the group-count term as one variable, target 7.5 (6.49 objects plus one
background group), and reads `groups_per_image` as well as the three metrics.
