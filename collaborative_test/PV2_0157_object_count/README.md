# PV2_0157 — object-count accuracy: the head nearly halves it, and Tetrominoes is a total failure

**Status** completed. 320 images per dataset, one core seed, head over 5 seeds.
**Parent** `PV2_0156`, which needed a measurement no existing metric provides.

**Why.** Every metric in this project scores the *partition*. None answers "how many objects did
it think there were." That became unavoidable in `PV2_0156`: the core whose predicted count is
nearly exact on CLEVR has the **worst** matched-object IoU, and the one that over-segments most
has the **best**. So the count is not a proxy for anything already measured.

**Definition.** Counts are read exactly as the contract reads a prediction: the number of
distinct non-zero labels in the grid. Background is id 0 on both sides and is not an object.
`exact` is the fraction of images whose predicted count equals the true count.

## True object counts, and a correction

| dataset | true objects / image |
|---|---|
| CLEVR fixed320 | **6.36** (min 2, max 10) |
| Tetrominoes test | **3.00** (exactly 3, every image) |
| Multi-dSprites test | **2.49** (min 1, max 4) |

**`PV2_0152` and `PV2_0153` both give CLEVR's truth as "~3.0".** That is wrong — 3.0 is
Tetrominoes' figure, carried across by mistake. CLEVR's is 6.36. Both files are corrected, and
the correction matters: it reverses the reading of their region-count tables (below).

## Result

| core | dataset | truth | pred | exact | within ±1 | MAE | bias |
|---|---|---|---|---|---|---|---|
| base 253 | CLEVR | 6.36 | 8.04 | 0.175 | 0.487 | 2.88 | +1.68 |
| base 253 | **Tetrominoes** | 3.00 | **21.07** | **0.000** | 0.003 | 18.07 | +18.07 |
| base 253 | Multi-dSprites | 2.49 | 6.62 | 0.281 | 0.469 | 4.56 | +4.13 |
| cont5000(mdS) | CLEVR | 6.36 | 9.90 | 0.144 | 0.372 | 4.20 | +3.54 |
| cont5000(mdS) | Multi-dSprites | 2.49 | 7.86 | 0.281 | 0.441 | 5.70 | +5.37 |
| **cont5000(Tetro)** | **CLEVR** | 6.36 | **6.65** | 0.184 | **0.500** | **1.99** | **+0.29** |
| cont5000(Tetro) | Tetrominoes | 3.00 | 17.94 | 0.003 | 0.003 | 14.94 | +14.94 |

With the fixed label-free head, 5 seeds, mean:

| core | dataset | pred | exact | MAE |
|---|---|---|---|---|
| base 253 | CLEVR | **12.90** | 0.035 ± 0.012 | 6.59 ± 0.32 |
| base 253 | Tetrominoes | 25.00 | 0.000 | 22.00 ± 0.66 |
| base 253 | Multi-dSprites | 10.61 | 0.044 ± 0.006 | 8.25 ± 0.33 |
| cont5000(mdS) | CLEVR | 15.19 | 0.006 ± 0.003 | 8.84 ± 0.45 |
| cont5000(mdS) | Multi-dSprites | 11.69 | 0.037 ± 0.023 | 9.33 ± 0.96 |
| cont5000(Tetro) | CLEVR | 11.38 | 0.051 ± 0.006 | 5.14 ± 0.37 |
| cont5000(Tetro) | Tetrominoes | 22.69 | 0.000 | 19.69 ± 1.01 |

## Three findings, one of which is bad news

**1. The head buys its metric gains by splitting, and object counting pays.** It raises the
predicted count on **every row without exception** — CLEVR 8.04 → 12.90, Tetrominoes 21.07 →
25.00, Multi-dSprites 6.62 → 10.61 — and cuts `exact` by a factor of three to five (CLEVR 0.175 →
0.035). The head's contribution to FG-ARI, Foreground IoU and matched-object IoU is +0.003 to
+0.026 (`PV2_0146`, `PV2_0153`), and this is what it costs. **None of the three contract metrics
can see it**, which is the strongest argument in this project for carrying a fourth diagnostic.

**2. Tetrominoes is a complete failure at the thing the name suggests.** It is our best FG-ARI
cell (0.8580) and the predicted count is **21.07 against a truth of exactly 3, with `exact`
precisely 0.000 — not one image of 320.** The patch grouping is right and the object count is
never right. Any sentence of the form "the model finds the objects on Tetrominoes" is false as
stated; "the model groups the patches correctly on Tetrominoes" is true.

**3. Count accuracy and matched-object IoU run in opposite directions.**

| CLEVR | pred count (truth 6.36) | MAE | matched_object_iou |
|---|---|---|---|
| cont5000(Tetro) | **6.65** | **1.99** | **0.6207** (worst) |
| base 253 | 8.04 | 2.88 | 0.6410 |
| cont5000(mdS) | 9.90 | 4.20 | **0.6671** (best) |

Monotone, and inverted. **This refutes `PV2_0152`'s mechanism directly.** That record attributed
its CLEVR matched-object IoU loss to over-merging — objects joined to each other. But its
predicted count is 6.65 against a truth of 6.36, the most accurate of the three: it is not
merging objects away, it is getting the number nearly right and still losing the metric.
`PV2_0153` withdrew that mechanism using a region-count table that itself relied on the wrong
truth (3.0); this is the measurement that settles it, and the withdrawal stands on better
evidence now.

## Why this is the expected behaviour of this architecture, not a bug

`model.py` and `loss.py` contain **no decoder and no reconstruction term** — zero matches for
`decoder`, `reconstruct`, `deconv`, `upsample` — and the objective is
`primary + 5.*spike` over PLV structure and spike synchrony (`loss.py:21`). For variant 253
`self.head` is `None` as well (`model.py:57`).

Every baseline this project compares against reconstructs the image: Slot Attention's
spatial-broadcast decoder, the CAE family's complex-valued autoencoder MSE, AKOrN and ItrSA's
autoencoding. A reconstruction loss penalises splitting one object into two, because both halves
must then reconstruct the same texture independently. **Our objective contains no such pressure**,
so nothing stops the core from carving one tetromino into seven synchronised fragments — they can
all be mutually synchronous and the loss is satisfied. The count errors are a predicted
consequence of the design, not an implementation fault, and they are the price of the thing that
makes the comparison interesting.

## What this licenses

**Licenses:** "our readout groups patches well and counts objects badly, the head makes the
counting substantially worse, and on Tetrominoes the count is never right."

**Does not license** any claim that the model "detects" or "counts" objects. It also does not
license dropping the head: the head's gains on the three contract metrics are real and
reproducible over 5 seeds. It licenses **reporting both**, which is what the benchmark plan now
has to say.

**Limits.** One core seed. Counts come from `regions_253` and the head's `predict`, both of which
are the scored readout, so there is no separate counting mechanism being evaluated — this is the
count implied by the masks we report, which is the number that matters.

## Reproduce

```bash
C=/work/USERS/tkim1/sw0255b/benchmark_codes
python $C/objcount.py --cache <eval cache> --train-cache <train cache> \
  --dataset clevr --split fixed320 --code $C --config $C/configs/frontier.json \
  --run /work/USERS/tkim1/sw0255b/run --label base --out base_clevr.json
```

`objcount.py` is in `evaluation_howto/`. Omitting `--train-cache` reports the core alone.
