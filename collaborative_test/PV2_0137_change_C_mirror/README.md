# PV2_0137 — change C, finished: mirror invariance is harmful, not just unhelpful

**Status** completed
**Parent** PV2_0134 (change C, first attempt), PV2_0135 (which withdrew it)
**This closes the B → A → C sequence.**

## Why there was a second attempt

The first change C used the inference seed. PV2_0135 found the 253 rollout is bitwise
deterministic in it, so that objective compared a tensor with itself and measured nothing. The
result was withdrawn, and the open question was left open: *would* an invariance signal help the
head, if one existed?

A horizontal mirror is one that exists. The encoder is a learned CNN and is not
mirror-equivariant; the Kuramoto coupling carries a learned direction term over ordered node
pairs; the spike clustering then runs on genuinely different activity.

## The premise was verified first, this time

`cache_flip.py` rolls out `flip(image)` for all 256 training images per dataset and stores the
result in the mirrored frame. `chkflip.py` un-mirrors it and compares against the original run:

| | mean\|diff\| | relative | regions |
|---|---|---|---|
| img 3 | features 0.4346, gamma 0.2560 | **0.891** | 3 → **273** |
| img 24 | features 0.4273, gamma 0.2339 | **0.877** | 7 → **259** |
| img 29 | features 0.4342, gamma 0.2295 | **0.886** | 8 → **66** |

The two rollouts genuinely differ — by 89% of the feature magnitude. So the objective had
something real to measure.

The region counts are the warning sign, and they turned out to be the explanation: under
mirroring the core's spike-synchrony structure **fragments**, 3 coherent regions becoming 273
tiny ones. Foreground agreement after un-mirroring is only 0.91, 0.63 and 0.46 of patches on
those three images. The core is not merely non-equivariant; its grouping falls apart.

The experiment is built not to depend on that broken partition: both sides are given the
**same** candidate set — the original row's `masks`, `prior` and `allowed`, mirrored spatially,
which permutes the node axis and leaves the candidate axis untouched, so candidate *k* means the
same object on both sides. Only the activity differs. The mirrored logits are flipped back
before comparison.

## Result: every arm, every dataset, refuses to move

Six arms, 21 epochs, the collapse guards from the first attempt retained, rule B′ selecting:

| dataset | arms selecting epoch 0 | epochs passing the Foreground IoU floor |
|---|---|---|
| CLEVR | 6 of 6 | **1 of 21** |
| Tetrominoes | 6 of 6 | **0 of 21** |
| Multi-dSprites | 6 of 6 | **1 of 21** |

Not one trained epoch, under any weighting of mirror agreement against rgb and tv, survived the
Foreground IoU floor on Tetrominoes; one did on each of the others. Every arm fell back to epoch
0, the head before any update.

So the finding is stronger than "it does not help": **training on mirror agreement destroys
figure-ground separation within a single epoch.** Forcing the head to read mirrored activity the
same way drags it toward the degraded reading, where background is not separable — exactly what
the 3 → 273 fragmentation predicts.

The selected rows are therefore all epoch 0, which is the untrained head and already recorded in
PV2_0134; they are reproduced here only to show the selection behaved:

| dataset | | fg_ari | fg_iou | obj_iou |
|---|---|---|---|---|
| CLEVR | 253 | 0.8421 | 0.7545 | 0.6410 |
| | epoch 0 | 0.8517 | 0.7530 | 0.6575 |
| Tetrominoes | 253 | 0.8580 | 0.7885 | 0.8173 |
| | epoch 0 | 0.8590 | **0.6744** | 0.8214 |
| Multi-dSprites | 253 | 0.6974 | 0.8177 | 0.7628 |
| | epoch 0 | 0.7087 | 0.8169 | 0.7834 |

## What C settles

**The head direction is open, and invariance is not the way through it.** What works is what
PV2_0133–PV2_0136 established: change A (background as a real candidate), rule B′ (validation
FG-ARI under a Foreground IoU floor), and a fixed label-free recipe. What does not work, now
tested twice and on a verified premise the second time, is asking the head to be invariant —
because the core it reads is not.

That is also a fact about the core worth keeping: **its grouping is strongly
orientation-dependent.** A mirrored CLEVR scene is not out of distribution for the images, but
it is for this core's dynamics. If the core is ever retrained, mirror augmentation is a concrete
thing to try, and this is the measurement to compare against.

## Method note

Verifying the premise before trusting the experiment is the lesson of PV2_0135, and it is now
the first step of the procedure: `chkflip.py` ran before `fitC.py` was believed. The first
change C would have been caught in one minute by the same check.
