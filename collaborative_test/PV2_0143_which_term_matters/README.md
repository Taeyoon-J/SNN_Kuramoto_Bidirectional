# PV2_0143 — `positive_spike`, not `phase_primary`, is the term that tracks performance

**Status** completed (scores already on disk; no new training)
**Answers** the question PV2_0142 left open, and answers it against both of my readings
**Also** the first real seed spread this project has measured

## The question

PV2_0142 found that on Tetrominoes and Multi-dSprites, 253's `phase_primary` never moves —
eight runs, at most 7% and often the wrong way, against CLEVR's −47.0%. I offered two readings
and said the scores would decide:

- **(a)** the recipe fails on these datasets, so the cores are weak;
- **(b)** there is nothing to learn on that term, so the cores are fine.

The scores were already on disk. The six finished 3,500 × 10 per-dataset cores were evaluated
on 1,000 images of their own dataset, and `result.json` was never read.

## The answer: neither reading, globally. It splits by dataset.

| core trained on | `phase_primary` | `positive_spike` | fg_ari | foreground_iou | matched_object_iou |
|---|---|---|---|---|---|
| CLEVR | **−47.0%** | **−83.2%** | — | — | — |
| **Tetrominoes** | ~0% | **−23% to −43%** | **0.7436 ± 0.0465** | 0.7425 ± 0.0433 | 0.7510 ± 0.0279 |
| **Multi-dSprites** | ~0% | **−2.6% to −11.6%** | **0.2789 ± 0.0813** | 0.2232 ± 0.0549 | 0.1414 ± 0.0550 |

Per-seed values: Tetrominoes fg_ari 0.7491 / 0.7871 / 0.6946; Multi-dSprites 0.2636 / 0.3668 /
0.2063.

**Tetrominoes is reading (b) and Multi-dSprites is reading (a).** A flat `phase_primary` is not
a failure signal, and Tetrominoes is the counterexample: that term never moved and the core
still reaches 0.74 on all three metrics.

**What tracks performance is `positive_spike`.** CLEVR drops it 83% and is strongest;
Tetrominoes drops it 23–43% and reaches 0.74; Multi-dSprites drops it 2.6–11.6% and stays at
0.28. That ordering is the only consistent account of the three datasets, and it is the term
whose gradient path runs to the spikes the score is actually read from.

So the answer to "does our core only work on CLEVR" is **no**. Trained on Tetrominoes it works.
Multi-dSprites is the dataset where this recipe does not get off the ground, and the symptom is
specific: the spike term does not descend.

## Protocol caveat, which matters

These are **pixel-protocol** numbers (`result.json` exposes only `/pixel/...`), on **1,000**
images, from a **3,500 × 10** core. They must **not** be compared against our patch-protocol
figures (Tetrominoes 0.8580, Multi-dSprites 0.6974, 320 images, CLEVR-trained core at
10,000 × 10). The pixel protocol's ceiling is below 1.0 by construction; the patch protocol's
is 1.0.

What *is* valid is the comparison **between** these two rows: same protocol, same budget, same
image count, three seeds each. Tetrominoes 0.7436 against Multi-dSprites 0.2789 is a 2.7×
gap, far outside standard deviations of 0.047 and 0.081.

## The first measured seed spread, and what it does to our earlier numbers

| | fg_ari std | foreground_iou std | matched_object_iou std |
|---|---|---|---|
| Tetrominoes, 3 seeds | **0.0465** | 0.0433 | 0.0279 |
| Multi-dSprites, 3 seeds | **0.0813** | 0.0549 | 0.0550 |

PV2_0141 predicted our seed-0 deltas would not survive a real spread, using epoch-to-epoch and
arm-to-arm wobble as a lower bound. This is the actual thing, and it is much larger:

- Our reported Tetrominoes head gain was **+0.0006** FG-ARI. The seed standard deviation on
  that dataset is **0.0465** — **77× larger**.
- Our reported Multi-dSprites head gain was **+0.0067**. The seed standard deviation is
  **0.0813** — **12× larger**.

These stds come from a different core budget and protocol, so they are an indication rather
than the error bar for our numbers. But they are the right order of magnitude, and they say
plainly that **the Tetrominoes and Multi-dSprites head gains are not measurable at one seed.**
Only CLEVR's (+0.0086 / +0.0285 / +0.0298) are large enough to plausibly survive, and that
remains unproven until CLEVR seeds 1 and 2 exist.

## Consequence for PV2_0138, now running

The two cores being trained at 10,000 × 10 are at 66–69%, with `positive_spike` at −33.3%
(Tetrominoes) and **−0.1%** (Multi-dSprites). On the evidence above that predicts a decent
Tetrominoes core and a poor Multi-dSprites one. The extra data (3,500 → 10,000) may change
that; the runs will say.

## Reproduce

```bash
python3 perds.py     # reads result.json from the six finished per-dataset runs
```
