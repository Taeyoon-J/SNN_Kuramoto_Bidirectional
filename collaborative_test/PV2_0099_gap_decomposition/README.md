# PV2_0099 — how much of the peer gap is the contract, and a prediction of mine that was wrong

**Status:** completed. **Parent:** `PV2_0046`, `PV2_0090`. **Contract:** four variants,
built here.
**Question:** `PV2_0046` measured one published checkpoint at 0.6195 under our contract
and 0.8946 under the peer's — 0.275 on the same model. How much of the peer comparison
is measurement rather than modelling?
**Method:** hold the model fixed and move only the contract. The same three Slot
Attention checkpoints, scored on a 2x2 of {grid 16, grid 32} x {whole 240x320 frame,
centre 240x240 crop}, each prediction against targets built on its own geometry.

## Ground truth, four variants, validation 300

| frame | grid | foreground | objects/image |
| --- | --- | --- | --- |
| whole 240x320 | 16 | 0.118 | 6.32 |
| whole 240x320 | 32 | 0.121 | 6.64 |
| crop 240x240 at (0,40) | 16 | **0.152** | **6.49** |
| crop 240x240 at (0,40) | 32 | 0.156 | 6.63 |

## Result

| contract | fg_ari | fg_iou | obj_iou |
| --- | --- | --- | --- |
| crop, grid 16 | 0.7195 | 0.1475 | 0.1083 |
| crop, grid 32 | 0.7234 | 0.1512 | 0.1053 |
| **whole, grid 16** | **0.8065** | 0.1157 | 0.1017 |
| **whole, grid 32** | **0.8134** | 0.1195 | 0.0972 |

| axis | effect on fg_ari |
| --- | --- |
| crop vs whole frame, grid 16 | **-0.0870** |
| crop vs whole frame, grid 32 | **-0.0900** |
| grid 16 -> 32, whole frame | +0.0069 |
| grid 16 -> 32, crop | +0.0040 |

## The prediction that was wrong

I argued in conversation that a centre crop **makes the measurement easier**, because it
cuts objects off at the edges. It does the opposite: cropping costs 0.087.

The reason is visible in the ground-truth table. At the same grid count the cropped
region has **smaller cells** — 15x15 px against 15x20 — so small objects that majority
voting used to swallow now survive. Objects per image rise 6.32 -> 6.49 and foreground
0.118 -> 0.152. More objects is harder, and that outweighs whatever the lost edge objects
were worth.

This also closes a loose end in `PV2_0090`, which attributed +0.0513 of its +0.1446 to
misalignment and the rest to "the model doing better when it can see the whole scene".
That remainder was ~0.093; measured directly here on self-consistent geometries it is
**0.087**. The two numbers agree, so `PV2_0090`'s split of its effect stands.

## The number that matters for our own work

| | grid 16 -> 32 |
| --- | --- |
| Slot Attention (here) | **+0.0069** |
| ours (`PV2_0094`, matched extent) | **-0.0946** |

**The same change in grid is nearly free for Slot Attention and costs us 0.095.** A
difference of 0.10 on one axis, between two models on the same images.

This is `PV2_0095`'s over-segmentation from another angle. Slot Attention's group count is
**fixed by construction** — seven slots, whatever the resolution. Ours is **emergent**, and
when the grid gets finer it emerges at 19.5 groups against 6.49 true objects. Resolution
robustness is not a property either model has or lacks in general; it follows from whether
the number of groups is a parameter or an outcome. `PV2_0097` is testing exactly that term.

## Correction: the decomposition arithmetic below was built on a misreading

The table below uses "label rule (`PV2_0046`) +0.19". **That is wrong.** In `PV2_0046`
0.19 is a **foreground fraction** — the prevalence a peer-like `min_frac 0.10` rule
produces, 0.188, against contract v1's 0.1146 — not a score effect. I read a prevalence
as a delta and put it in a sum.

Worse, `PV2_0046` had already **tested and refuted** the hypothesis that a looser label
rule explains the peer's lead. Rebuilding targets at `min_frac 0.25` moves our own score
**0.7239 -> 0.5355** on fg_ari while foreground IoU barely moves, 0.7251 -> 0.7315,
because the patches a looser rule adds are partial-coverage patches at object boundaries
— exactly where the grouping is wrong. A looser rule makes us worse, not better.

So the label-rule row is not +0.19, it is **negative and already measured**, and the sum
below does not mean what it says. What the two axes measured here establish stands on its
own: geometry is **-0.083** and mask downsampling is **+0.0101** (`PV2_0100`), so the two
contract axes we can control account for about -0.07, and the 0.275 is **not** explained
by anything tested so far.

## The decomposition does not close

| component | fg_ari |
| --- | --- |
| geometry (crop + grid), measured here | **-0.083** — our contract is *easier*, not harder |
| label rule (`PV2_0046`) | +0.19 |
| sum | +0.107 |
| **observed cross-contract difference** | **+0.275** |
| **unexplained** | **~0.168** |

So the peer's contract is still easier than ours, but **not for the reason I gave**, and
over half of the difference is unaccounted for.

The axis not measured here is the one most likely to hold it: their ground truth is
derived from **masks downsampled to 128x128** before cell voting, while ours votes on the
native 240x320 masks. Downsampling masks deletes small objects before they can be
counted, which lowers objects per image and makes every metric easier — the same
mechanism that made the *crop* harder, running the other way. Testing it needs only our
own masks and is cheap; it is not run here because this experiment was scoped to the two
axes the peer's records actually document.

## Limitations

Three seeds, validation 300, one seeded inference draw each. Slot Attention only — the
same 2x2 on our own model needs a 32-grid checkpoint per cell and was not run. The peer's
contract is **not** reproduced: their records give the pool, the IDs and the cell rule but
never the crop geometry, so "peer-like" here means our best reconstruction of the two
axes they document.
