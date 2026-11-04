# PV2_0133 — what the readout can and cannot buy, and what the benchmark can honestly claim

Status: **in progress.** The 252 diagnosis, the quantisation ceilings and the Slot Attention
comparison are complete; the A+B readout result is running.

## 1. Slot Attention, on our images, with our evaluation function

Previous comparisons put their published 98.8 beside our patch score. Those are different
renders, different object counts and different protocols, so the pair never meant anything.
This runs the published `ckpt-500` over the **same 320 images** (fixed320, ids 1320-1639,
from `images40k`), maps its slot masks to the **same grid** by the same majority rule, and
scores with the **same `evaluate_patch_masks` call** against the **same targets**.

| | fg_ari | fg_iou | obj_iou | predicted foreground |
| --- | ---: | ---: | ---: | ---: |
| **253 (ours)** | **0.8421** | **0.7545** | **0.6410** | 0.2001 |
| Slot Attention, grid 64 | 0.3808 | 0.2022 | 0.1053 | **0.8510** |
| Slot Attention, grid 32 | 0.3912 | 0.2028 | 0.1070 | 0.8450 |
| truth foreground | | | | 0.2115 |

**The grid is not what holds Slot Attention back**: 32 and 64 give the same answer. It calls
**85% of the image foreground against a 21% truth**, and that single failure drags all three
metrics. Our own earlier baseline saw the same 85.3%.

**Stated rather than buried:** this split is `clevr_with_masks`, not the original CLEVR
render the authors trained on, and it is CLEVR10 rather than CLEVR6; their README says the
two renders are not directly comparable. It is one published checkpoint, not three seeds.
The honest sentence is *"on identical data and identical scoring we lead, and this data is
not the render Slot Attention was trained on"* -- not *"we beat Slot Attention"* unqualified.

## 2. The quantisation ceiling decides which datasets are winnable

A grid-`g` prediction cannot resolve anything finer than a cell. Quantising the pixel truth
to the grid and scoring it against itself is therefore a hard upper bound, and it is a
property of the data, not of any model.

| dataset | native pixels | grid 32 | grid 64 | Slot Attention published |
| --- | --- | ---: | ---: | ---: |
| CLEVR | 128x128 | 0.8823 | **0.8988** | 0.988 (CLEVR6) |
| Tetrominoes | **35x35** | **0.9352** | impossible | 0.995 |
| Multi-dSprites | **64x64** | 0.6017 | **1.0000** | 0.913 |

Against their **published pixel numbers**, CLEVR and Tetrominoes are unwinnable at any grid
we can run -- Tetrominoes has only 35 pixels per side, and CLEVR would need grid 128, whose
`direction_learner` is 16,384² = 268M parameters, 16x grid 64's memory. **Multi-dSprites is
the exception: ceiling 1.0000, and their weakest published result.**

That is the argument for scoring both models on the same grid instead, which our contract
already requires -- and section 1 is that comparison.

## 3. Why 252 lost to 253 on identical data

Same 10,000 images, same 6,250 updates, same grid. 252 is **supervised** and finished at
0.5213 against 253's 0.8421 on the patch contract.

**252's core did not learn.**

| | 253 | 252 |
| --- | ---: | ---: |
| `phase_primary` | 4.20 -> 2.28 (**-46%**) | 7.30 -> 7.29 (**-0.1%**) |
| `positive_spike` | 15.46 -> 2.50 (**-84%**) | 15.84 -> 10.38 (-34%) |
| `final_mask` (252 only) | — | 9.22 -> 3.99 (-57%) |

The head learned well; the core stood still. And it stood still **while receiving more
gradient**:

| parameter | change 253 -> 252 | gradient sum 253 -> 252 |
| --- | --- | --- |
| `gamma_channel_proj.weight` | 0.111 -> **0.034** | 25,220 -> **202,842** |
| `log_temperature` | 0.108 -> **0.0096** | 285 -> 763 |
| `dendric_layer.dense.bias` | 0.103 -> **0.0137** | 9,017 -> 32,247 |
| `direction_learner` | 0.00215 -> **0.00015** | 19.9 -> 24.7 |

**More gradient and less movement is cancellation.** The mechanism is the path length. 252's
body loss is `core_objective + teacher_objective`, and the two reach gamma by very different
routes:

```
core objective     gamma -> 1,024-step rollout -> spikes -> loss   (only the last 64 differentiable)
teacher objective  gamma -> loss                                    (direct)
```

The short path wins, and what it optimises is gamma-frequency agreement, not synchrony. The
mask loss itself never reaches the body at all -- `features.detach()` for variant 252, which
the binding records as `final_mask_to_body_gradient: false`. So the only channel by which
ground-truth masks touch the core is the teacher term, and that term **erases the core's own
learning signal rather than adding to it**.

**This is why supervision lost to no supervision**: the supervised variant spends its
gradient budget cancelling the objective that actually builds synchrony.

The curve confirms it is not merely undertrained in the usual sense -- it was still rising
(u3000 0.4454 -> u6250 0.5239 on 160 fixed320 images) -- but `foreground_iou` sat at 0.2021
at **every** checkpoint, because the head designates no background. That metric cannot move
with more budget.

## 4. What the readout can buy, measured

`min_group_size` was 32 by default and had never been swept at grid 64. Selected on CLEVR
validation (0.8391 -> 0.8501) and then read once on fixed320:

| | min32 | **min8** |
| --- | ---: | ---: |
| fg_ari | 0.8332 | **0.8421** |
| foreground_iou | 0.7433 | **0.7545** |
| matched_object_iou | 0.5990 | **0.6410** |

All three rose from one threshold. Predicted foreground moved 0.1982 -> 0.2001 against a
0.2115 truth.

Threshold sensitivity across datasets, which is the premise behind adding a learned head:

| | min8 vs min64 fg_ari spread |
| --- | ---: |
| CLEVR | 0.0550 |
| Tetrominoes | 0.0030 |
| Multi-dSprites | 0.0028 |

Only CLEVR is sensitive, and min8 is best or tied everywhere, so the thresholds do not in
fact need per-dataset hand tuning.

## 5. Generality: a CLEVR-only core on datasets it never saw

The core trained on CLEVR alone, frozen, scored zero-shot:

| dataset | fg_ari | fg_iou | obj_iou | conditions |
| --- | ---: | ---: | ---: | --- |
| CLEVR | 0.8421 | 0.7545 | 0.6410 | fixed320, 320 images, patch64 |
| Tetrominoes | 0.8580 | 0.7885 | 0.8173 | TEST 320, zero-shot |
| Multi-dSprites | **0.6893** | **0.8193** | **0.7540** | TEST 1000, **pixel**, zero-shot |

Multi-dSprites is the one to note. Our records had it as the standing weak point -- 0.2817
in our own pipeline, 0.2789 ± 0.0813 over three seeds in the peer engine at grid 32 trained
*on Multi-dSprites*. A CLEVR-only grid-64 core reaches **0.6893 on the same pixel protocol
and the same 1000 TEST images**, having never seen the dataset. The protocol explanation I
expected -- that the patch grid flatters a sparse 8.65% foreground -- did not survive: the
pixel number is 0.6893, not a quantisation artefact.

What still differs between 0.6893 and 0.2789 is the grid (64 vs 32) and the training data
(CLEVR vs Multi-dSprites). Which of the two carries the gain is not yet separated.

## Limits

Every number above is **one seed**. Tetrominoes and Multi-dSprites are zero-shot from a
CLEVR core. Slot Attention is one published checkpoint on a render it was not trained on.

## 6. The head direction is open — two fixes, both from measurements

Yesterday the same learned readout scored **0.8197** on CLEVR fixed320 against 253's 0.8421,
and its Foreground IoU was pinned at the truth foreground fraction. Two changes, each
dictated by a measurement rather than guessed, reverse that.

**A. Give the head a background candidate.** It had been fed `regions_from_output` — the
all-ones whole-image partition — so background was never among the candidates and `argmax`
could not pick it. `candidates()` already admits label 0 as an ordinary region ("Include ID
0 as an ordinary region too"), so feeding **253's own partition** supplies background for
free; the only other change is undoing the `+1` offset for that candidate on output. The
head now starts from 253's answer and learns corrections, so it cannot collapse below it by
construction.

**B. Select arms on validation FG-ARI, not validation RGB MSE.** On CLEVR the five arms'
RGB MSE ranked *exactly opposite* to their FG-ARI — a clean five-point inversion — so
RGB-MSE selection chose the worst arm available. Our contract selects hyperparameters and
thresholds on validation, labels included; what it forbids is selecting on the reserved
evaluation split.

### CLEVR, fixed320, 320 images, grid 64

| | fg_ari | fg_iou | obj_iou |
| --- | ---: | ---: | ---: |
| 253 | 0.8421 | 0.7545 | 0.6410 |
| **head v2** | **0.8503** | **0.7837** | **0.6706** |
| delta | **+0.0082** | **+0.0292** | **+0.0296** |

Patches changed: **2.04%**. The head is not overturning 253's partition; it is repairing a
small number of misassignments, which is what A was designed to allow.

Validation FG-ARI per arm, against the 0.8501 baseline:

| arm | val fg_ari | |
| --- | ---: | --- |
| `local_rgb_tv` | **0.8602** | selected |
| `local_rgb_r2` | 0.8585 | |
| `local_rgb_spike` | 0.8553 | |
| `local_rgb_trust` | 0.8539 | |
| `local_rgb` | 0.8479 | the only arm below baseline |

**Four of five beat the baseline, and the one that fails is pure RGB reconstruction.** The
arms that win all add a signal that is not RGB: `tv` weights the mask's total variation by
colour edges, `spike` ties neighbouring assignments to the core's own temporal co-firing.
That matches what the five-point inversion already implied — RGB reconstruction alone
points away from object grouping; it only helps when something else supplies the direction.

## 7. The head direction is open on all three datasets

Core trained on **CLEVR only**, frozen. The head retrained per dataset on 256 of that
dataset's images with **no labels**, six minutes each, selected on that dataset's validation
FG-ARI. Peer patch contract, 320 evaluation images, one seed.

| dataset | readout | fg_ari | fg_iou | obj_iou | patches changed |
| --- | --- | ---: | ---: | ---: | ---: |
| **CLEVR** | 253 | 0.8421 | 0.7545 | 0.6410 | |
| | **head v2** | **0.8503** | **0.7837** | **0.6706** | 2.04% |
| | delta | **+0.0082** | **+0.0292** | **+0.0296** | |
| **Tetrominoes** | 253 | 0.8580 | 0.7885 | 0.8173 | |
| | **head v2** | **0.8586** | **0.7895** | **0.8187** | 0.57% |
| | delta | **+0.0006** | **+0.0010** | **+0.0015** | |
| **Multi-dSprites** | 253 | 0.6974 | 0.8177 | 0.7628 | |
| | **head v2** | **0.7041** | **0.8213** | **0.7660** | 0.33% |
| | delta | **+0.0067** | **+0.0036** | **+0.0032** | |

**Nine of nine cells improve.** The head changes 0.33–2.04% of patches: it repairs a small
number of misassignments rather than overturning 253's partition, which is what change A
was built to allow.

The gain tracks where there is something to repair. CLEVR, whose threshold sweep spanned
0.0550 between `min_group` 8 and 64, gains 0.0082; Tetrominoes and Multi-dSprites, whose
sweeps spanned 0.0030 and 0.0028, gain 0.0006 and 0.0067.

**Generality, which was the point of the exercise:** the core never saw Tetrominoes or
Multi-dSprites. Only the head was swapped, trained unsupervised on the target dataset in
minutes. The arm chosen differs per dataset — `local_rgb_tv` on CLEVR, `local_rgb_trust` on
the other two — which is exactly the per-dataset adaptation a learned head is supposed to
provide and a hand-set threshold is not.

## 8. The ceiling claim in section 2 applies to the pixel protocol only

Section 2's ceilings (CLEVR 0.8988, Tetrominoes 0.9352) are for the **pixel** protocol,
where the prediction is expanded back to 128×128 and scored against the original pixel
truth. **Under the patch contract the ceiling is 1.0000 by construction** — the target is
the truth quantised to the same 64×64 grid, so a perfect prediction matches it exactly.

That is the contract this section and the Slot Attention comparison both use, and it is the
one the benchmark should report: it removes the grid from the comparison instead of letting
it cap one side. The earlier conclusion that "CLEVR and Tetrominoes cannot be won" holds
only against Slot Attention's *published pixel* numbers, not against Slot Attention scored
on our grid.

## 9. Why Slot Attention drops from 0.8102 to 0.38 — not the grid

Two things changed between the old 0.8102 and yesterday's 0.3808, not one. Holding the
split fixed and moving only the grid:

| grid, all on fixed320 | fg_ari | predicted foreground |
| ---: | ---: | ---: |
| 16 | 0.4153 | 0.8513 |
| 32 | 0.3912 | 0.8450 |
| 64 | 0.3808 | 0.8510 |

**0.035 across a 4× change of grid.** The 0.4 gap is therefore almost entirely the
evaluation set and targets — the old number was our own split (test ids 7000+) scored
against `targets_v1.pt`, the new one is fixed320 (ids 1320–1639) scored against the peer
engine's `labels64`.

What does not change with the grid is the failure: at every grid Slot Attention calls
**85% of the image foreground against a 21% truth**, and the slot designated background
covers only 14.9% of patches. Its background is split across several slots, so no single
slot dominates and the "largest slot is background" rule cannot find it. The old run
measured the same 85.3%.

A remaining candidate for part of the gap, not yet tested: the background rule itself. Our
runner designates the slot covering the most patches overall; the peer's Slot64 contract
designates the slot covering most of the one-pixel border. With a fragmented background
those pick different slots. Both rules can be applied to the predictions already saved, at
no GPU cost.
