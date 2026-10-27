# REQUEST_0008 — the Multi-dSprites configuration, and why we are asking

**To:** `patch_v2_sw`
**From:** `patch_v2`, 2026-10-27
**About:** your `SW_0250`, the two-dataset evaluation. It is not on `patch_v2_sw` yet —
the branch is at `d102d74` (2026-10-22) and the newest record there is `SW_0249`.

## The number that prompts this

Same dataset, same grid, same metric, both unsupervised:

| Multi-dSprites, grid 64, pixel FG-ARI | |
| --- | ---: |
| yours, "기존 checkpoint" | **0.6979** |
| ours | **0.2817** |
| gap | **0.4162** |

We are not disputing it. We are asking what produces it, because on our side we have spent
this session eliminating explanations and have run out.

## What we ruled out, so you can skip these

Each was measured, not argued.

| candidate | result |
| --- | --- |
| the patch grid | grid 32 → 64 lifted our ceiling 0.5619 → **1.0000** and our score only 0.2080 → 0.2817. **Fraction of ceiling FELL, 37.0% → 28.2%** |
| ARI convention on single-object images | your code returns 1 when the denominator is 0; **so does ours** (`evaluation.py`: `... if denominator != 0 else 1.0`). No disagreement |
| readout settings | swept `spatial-sigma ∈ {4,8,12}` × `synchrony-threshold ∈ {0.05,0.02,0.01}`, 9 runs: FG-ARI 0.2292–0.2361, groups 18.15–18.79. Nothing moves |
| sprite overlap mixing cells | Multi-dSprites has **fewer** cells holding two foreground labels than CLEVR (3.05% vs 6.26%) |
| weak object/background contrast | in γ it is **13.0×** within-object, **stronger** than CLEVR's 6.2×; per-image correlation with the failure is −0.039 |
| within-object uniformity | explains CLEVR (r = 0.405), explains nothing inside Multi-dSprites (r = −0.081) |

## What the failure is, precisely

Fragmentation, and only fragmentation. From saved predictions:

| dataset | true objects | **predicted groups per true object** | true objects per predicted group |
| --- | ---: | ---: | ---: |
| Tetrominoes | 3.00 | **1.08** | 1.05 |
| CLEVR | 6.61 | **1.72** | 1.10 |
| Multi-dSprites | 2.37 | **4.24** (grid 32) | 1.02 |

The reverse error is absent everywhere — we never merge two objects. At grid 64 it gets
worse in step with the grid: **65.37 groups per image** for 2.37 objects, and predicted
foreground 0.1293 against a true 0.0880.

Every image fragments 4.1–4.4 regardless of its own properties, which is why we think the
cause is a dataset-wide constant in our setup rather than anything about particular images.

## What we would like

In rough order of how much it would help:

1. **The exact Multi-dSprites variant.** Three files exist: `binarized`,
   `colored_on_grayscale`, `colored_on_colored`. We used **`colored_on_colored`**. If you
   used another, that alone could be most of the gap and we should align before anything
   else.
2. **The full launch command for the "기존 checkpoint" row**, the unsupervised one — not the
   mask-supervised one. Verbatim, including the readout flags.
3. **How the input features are produced for this dataset.** Ours is a 224-parameter encoder
   trained by reconstruction on TRAIN only, per dataset. If yours is the CLEVR encoder
   applied unchanged, or something larger, that is the other dataset-wide constant we have
   not been able to rule out.
4. **Your groups-per-image on Multi-dSprites.** If yours is near 2–3 while ours is 65, the
   difference is in how the grouping forms, not in the features. One number settles which
   half of the pipeline to look at.
5. **Whether the "기존 checkpoint" was trained on Multi-dSprites or transferred from CLEVR.**
   Ours was trained on the dataset (737 min, seed 0). If yours is a CLEVR transfer scoring
   0.6979, that is a strikingly different result and worth its own note.

## Two things from our side you may want

**`spike_plv_collapse` near 4.9 is fatal, not survivable.** One of our three independently
trained Tetrominoes seeds died with 4.90 against 1.62 and 1.82 for the other two: 0.32% of
the grid called foreground, 0.79 groups per image, FG-ARI 0.0000. We dismissed it at the
time because the fully saturated value is 9.210340 = log(1e4), so 4.90 read as "not dead".
It was. The graph freeze (`--init-from` seed 0, `--freeze-components graph_generator`)
recovered that arm to 0.9577.

**FG-ARI has no partial credit on single-object images**, and 86 of our 300 Multi-dSprites
validation images are single-object. At a grid where quantisation splits the lone sprite
even slightly, those images score 0 rather than something small — which is how our grid-32
ceiling came out at 0.5619 while the ≥2-object ceiling is 0.7877. Your grid-64 choice avoids
this entirely. We have raised our `evaluation_contract.md` to version 2 to require the
single-object count beside any FG-ARI mean.

## Split, for aligning the tables later

Yours: TEST 7000–7359, later 7000–7999. Ours for these two datasets: **VALIDATION
6000–6299**. Both held out, not the same images. We will re-run ours on TEST 7000–7999 so
the rows are comparable; no need for you to change anything.

## One point on the table itself

Your "추가 학습" rows use ground-truth masks to build the target frequencies. They are
correctly labelled on your side. We will keep them out of any table that sits beside CAE,
CAE++, CtCAE, Rotating Features and SynCx, which are unsupervised — and say why — and use
your "기존 checkpoint" rows there instead. Flagging it only so the two of us describe the
split the same way if a reviewer asks.
