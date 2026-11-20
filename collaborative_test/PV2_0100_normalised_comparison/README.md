# PV2_0100 — mask downsampling is worth 0.003, and the normalised comparison flips

**Status:** completed. **Parent:** `PV2_0099`, `PV2_0046`, `PV2_0090`.
**Two results: one measurement, one correction to a conclusion this project has been
carrying since `PV2_0046`.**

## 1. Voting on downsampled masks changes almost nothing

`PV2_0099` left 0.168 of the 0.275 cross-contract difference unexplained and named the
likeliest remaining axis: the peer's ground truth comes from a 128x128 HDF5, so instance
masks were downsampled before any cell voting, while ours votes on native 240x320 masks.
The expectation was that downsampling deletes small objects before they can be counted.

Same Slot Attention predictions, targets differing only in that step. Nearest-neighbour,
the only correct resize for an integer label map.

| ground truth | foreground | objects/image | fg_ari |
| --- | --- | --- | --- |
| native 240x320, grid 16 | 0.118 | 6.32 | 0.8065 |
| downsampled 128x128, grid 16 | 0.115 | 6.27 | 0.8093 |
| native 240x320, grid 32 | 0.121 | 6.64 | 0.8134 |
| downsampled 128x128, grid 32 | 0.116 | 6.64 | 0.8166 |

**+0.0028 and +0.0032.** Objects per image barely move — 6.32 to 6.27, and 6.64 to 6.64.
A 240x320 mask downsampled to 128x128 still resolves every object that survives cell
voting at these grids. The hypothesis was wrong, the second of mine in a row on this
question.

Running total: geometry **-0.083** (`PV2_0099`), downsampling **+0.010**, against an
observed **+0.275**. The two axes we can control account for none of it.

## 2. The correction that matters

`PV2_0046` established the framing this project has used since for cross-branch claims:

> Normalised to each branch's own Slot reference: this branch is at **1.17x** of Slot,
> the peer at **0.87x**. So in the only comparison that holds, this branch is ahead and
> theirs has not yet matched its reference.

That framing is right and the numbers in it are now **out of date**, because `PV2_0090`
changed our Slot reference. Ours was measured through the published centre crop, which
misaligned the prediction grid against the target grid; aligned, the same checkpoints
score 0.8102 instead of 0.6656.

| | model | own Slot reference | ratio |
| --- | --- | --- | --- |
| ours, as `PV2_0046` had it | 0.7603 | 0.6656 (crop) | **1.14x** |
| **ours, corrected** | **0.7603** | **0.8102 (aligned)** | **0.94x** |
| **peer** (SW0137/SW0138 vs their Slot32) | **0.7819** | **0.7817** | **1.00x** |

**The normalised comparison flips.** `PV2_0046`'s conclusion — that we lead on the only
comparison that holds — rested on a Slot reference that was depressed by a preprocessing
bug in our own harness. Corrected, **we are at 0.94x our reference and the peer is at
1.00x theirs.**

Two things keep this from being worse than it is, and neither rescues the fg_ari claim:

- The peer's 1.00x is a **tie with a margin of 0.0003** against a Slot32 seed spread of
  0.1353, which their own records decline to call a victory.
- Our two IoU results are untouched and are not close: foreground IoU **0.7132 against
  0.1151** and matched-object IoU **0.6598 against 0.1025**, on aligned predictions. Slot
  Attention predicts 85% of patches as foreground against an 11.8% target under every
  preprocessing tried.

## What this means for the goal

The honest position, replacing `PV2_0046`'s:

- **Absolute cross-branch numbers remain incomparable.** 0.275 separates the same
  checkpoint under the two contracts and the two axes we can measure account for none of
  it; the remaining difference is their split and render, which we cannot reproduce.
- **On fg_ari, normalised, we are behind** — 0.94x against 1.00x.
- **On figure/ground and per-object segmentation we are ahead by a factor of six**, and
  that is the part of this project that does not depend on a tie-break or a contract.

`ROUTE_MAP_0.9_v2.md` already says no demonstrated path to 0.9 exists. This does not
change that; it removes a comfort that was resting on our own harness bug.

## Limitations

Validation 300, three seeds, one seeded draw each, Slot Attention only. The peer's
ratio uses their reported SW0137/SW0138 and Slot32 numbers, read from their records and
not reproduced here; `PV2_0090`'s crop finding has not been checked against their Slot
baseline, and if it applies there too their reference moves as well.
