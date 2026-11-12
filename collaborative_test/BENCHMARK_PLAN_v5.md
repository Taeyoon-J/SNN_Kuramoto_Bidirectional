# Benchmark plan v5

> **Superseded by `BENCHMARK_PLAN_v6.md`** (after PV2_0154-PV2_0157). v5's protocol (§1) and
> its comparison design (§2: best-baseline-per-cell, narrowest cell, per-model budget
> verdict) stand unchanged. What v6 adds is a **fourth required metric, object count** --
> v5 does not mention it, and PV2_0157 showed the three contract metrics cannot see what
> the head costs there. v6 also adds: score a core on all three datasets before comparing
> it, T=512 as the default rollout, and the finding that out-of-domain data improves a
> dataset more than in-domain data does.

Supersedes v4. Written after PV2_0137–PV2_0151.

v4 was written when the reference column held **one** model, and that model was a published
checkpoint we could not fully trust. Since then four baselines have been trained on our own data
at our own budget and scored by our own function. That changes the design of the comparison, not
just its numbers, and this document is the revision.

Our position is unchanged: the claim is that this model finds objects, and Slot Attention is the
bar it has to clear. It now clears it by a wide margin under a protocol we control — and the
remaining weakness is no longer the reference column. **It is our own seed count.**

## 1. The protocol, unchanged from v3 and v4

- **Patch contract at grid 64.** Ground truth is quantised to the grid and scored there. The
  ceiling is 1.0 by construction. A predicted mask is never expanded to pixels to be scored.
- **One function, `evaluate_patch_masks`**, for every model, on the same image ids in the same
  order.
- **Three metrics, always reported together and never combined**: `patch_fg_ari`,
  `patch_foreground_iou`, `patch_matched_object_iou`.
- **The score comes from classifier masks on spikes.** `spike_synchrony_components` on
  `trace['spikes']`, then `foreground_induced_merge`, optionally refined by the readout head.
  theta and PLV are diagnostics and are never the score.

Two additions, both forced by scoring models that do not emit patch grids (`PV2_0150`):

- **A prediction coarser than the grid is lifted by `np.repeat`; a prediction finer than the
  grid is reduced by patch majority vote** (`to_grid64.py`), ties to the smaller label id. These
  are the same rule in two directions. Nothing is expanded to pixels either way.
- **The truth tensor is shared, not re-derived.** A baseline is handed
  `Dataset.labels(ids, split)` itself, lifted to the model's input resolution by an exact
  `np.repeat`. It must not be handed a separately-computed pixel mask, because then its own
  metric and ours are no longer measuring the same thing.

## 2. What v4 got wrong, and the four things that replace it

### 2.1 v4's biggest worry is resolved: Slot Attention's weak score is not domain shift

v4 §2.3 said *"Beating a model that is failing to transfer is not beating the model"* and
required a retrained row before any headline. **That row exists** (`PV2_0149`): Slot Attention
trained from scratch on our render, on our 10,000 CLEVR ids, at 100,032 exposures.

| Slot Attention on CLEVR fixed320 | fg_ari | fg_iou | obj_iou |
|---|---|---|---|
| published checkpoint, transferred | 0.3747 | 0.2219 | 0.1165 |
| **retrained here, exposure-matched** | **0.2872** | 0.2220 | 0.0801 |

Retraining it on our render does **not** rescue it, and 320× more data moves FG-ARI only to
0.3747 with Foreground IoU identical to four decimals. It uses 6.89–6.91 of 7 slots with
near-equal shares on both checkpoints: **it tiles rather than segments on these images
regardless of how long it trains.** v4's caveat was the right caution and the measurement
retired it.

`PV2_0151` finished the row: run zero-shot on the other two datasets it scores
**0.1739 / 0.2192 / 0.1071** and **0.1930 / 0.0876 / 0.0680** — worse off its training set than
on it, calling 84% and 85% of the image foreground. It is last in all nine cells.

**Replaces v4 §2.3's requirement.** The two-row reference table is no longer needed for Slot
Attention. What is still owed is 3 seeds and the published 500k-step budget (§4).

### 2.2 One reference model is the wrong design — report best-baseline-per-cell

v4 assumed a single reference column. `PV2_0151` shows that is not safe: **which baseline comes
second depends on the dataset.**

| | CLEVR | Tetrominoes | Multi-dSprites |
|---|---|---|---|
| **ours, 253 core, no head** | **0.8421 / 0.7545 / 0.6410** | **0.8580 / 0.7885 / 0.8173** | **0.6974 / 0.8177 / 0.7628** |
| AKOrN (conv) | 0.6046 / 0.4219 / 0.2769 | **0.7202 / 0.7540** / 0.5913 | 0.4055 / **0.6834** / 0.5434 |
| AKOrN (attn) | 0.6027 / 0.3327 / 0.3750 | 0.5676 / 0.7383 / 0.4687 | 0.4343 / 0.6019 / 0.5751 |
| ItrSA | 0.5794 / 0.2787 / **0.3853** | 0.6319 / 0.7137 / 0.5687 | **0.4817** / 0.2489 / **0.6093** |
| Slot Attention | 0.2872 / 0.2220 / 0.0801 | 0.1739 / 0.2192 / 0.1071 | 0.1930 / 0.0876 / 0.0680 |

Bold marks each cell's best baseline. Three different models own those cells; **no single
baseline is the right reference.** Picking one would let us choose our opponent.

**Requirement.** Every headline table carries a **best-baseline-per-cell** row, computed per
cell across every baseline we have run, and our margin is quoted against that composite — not
against any one model, and never against the field's average.

Under that rule we are first in **nine of nine**, with margins +0.1343 to +0.3326 — except one
cell, below.

### 2.3 Name the single narrowest cell, every time

**Tetrominoes Foreground IoU: 0.7885 against AKOrN (conv)'s 0.7540, +0.0345.** Every other
margin is 0.134–0.333. We have no seed spread for the conv variant, and `PV2_0150` measured
AKOrN (attn)'s own two-seed FG-ARI spread on Tetrominoes at 0.0535 — larger than this margin.

**Requirement.** The report states this cell as unsettled rather than letting "nine of nine"
stand unqualified. A second AKOrN (conv) seed costs 7 minutes and is the cheapest thing on the
queue that changes a conclusion.

### 2.4 The budget caveat is per-model, not global — so publish it per model

v4 had no way to say how much of a baseline's weakness was its budget. Now there is one: compare
each baseline **against its own published number**, which is a comparison of a model with itself
and so survives the protocol mismatch.

| model | published 3-seed FG-ARI (Tetro / dSprites / CLEVR) | ours at 1/210 budget | starved? |
|---|---|---|---|
| AKOrN (conv) | 76.4±0.8 / 63.8±7.7 / 59.0±4.3 | 72.1 / 40.7 / **60.5** | **no on CLEVR, barely on Tetrominoes** |
| AKOrN (attn) | 88.6±1.7 / 78.3±1.3 / 91.0±0.5 | 56.8 / 43.5 / 60.4 | yes |
| ItrSA | 85.8±0.8 / 68.1±1.4 / 82.5±1.7 | 63.2 / 48.3 / 58.1 | yes |
| Slot Attention | — | 0.2872 CLEVR | **no** — 320× more data buys +0.0875 |

**AKOrN (conv) at 1/210 of its budget already matches its published CLEVR number.** So on
Tetrominoes we are beating a model that is not starved, and the "give it its full budget"
objection does not cover that cell. For attn and ItrSA it does.

**Requirement.** Every baseline row carries its own published figure beside it and a one-word
verdict on whether that row is budget-limited. A claim that rests only on starved rows is
labelled as a **data-efficiency** claim; a claim that rests on an unstarved row is not.

## 3. Required diagnostics, promoted from footnotes

Three numbers turned out to carry more interpretive weight than any metric, and each is now
mandatory in the table rather than in prose.

### 3.1 Predicted foreground fraction against the truth's

| dataset | truth | ours | AKOrN attn | AKOrN conv | ItrSA | Slot Attention |
|---|---|---|---|---|---|---|
| CLEVR | 0.2115 | **0.2001** | 0.6140 | 0.4355 | 0.7218 | 0.7047 |
| Tetrominoes | 0.2368 | 0.2842 | 0.2800 | 0.2997 | 0.2787 | 0.8396 |
| Multi-dSprites | 0.0943 | **0.0841** | 0.1798 | 0.1070 | 0.4988 | 0.8486 |

This one column separates **tiling from segmenting** faster than any metric. Ours is the only
row that under-claims foreground on CLEVR and Multi-dSprites; every baseline over-claims by
2–3.4× on CLEVR. It is also the explanation for the Foreground IoU column, and it is why
Tetrominoes — where all four baselines are near the truth — is where our margin is thinnest.

### 3.2 Agreement between the baseline's own metric and ours

Each model's unmodified `eval_obj.py` FG-ARI against ours after the grid reduction agrees to
**≤0.0013 on every cell**, and exactly where no reduction is needed. That is what rules out "the
gap is your scorer".

**Requirement.** Any new baseline reports both numbers and their difference. If they disagree by
more than ~0.005, the conversion is wrong and the row does not ship.

### 3.3 Which background rule was selected, and on what

All four rules give the **identical FG-ARI** on every dataset and model we have tried, because
they only relabel patches the truth calls background and FG-ARI excludes those. So the rule
moves **Foreground IoU and matched-object IoU only** — which is exactly why it must be chosen on
validation (`bg_any.py`), never on the reported split.

**Requirement.** Every baseline row names its rule and the split that chose it.

## 4. Seeds — still the only thing standing between candidates and results

Unchanged from v4 in substance, worse in contrast: the reference column now has far more
measurement behind it than our own column does.

| | seeds |
|---|---|
| 253 core trained on CLEVR, grid 64 | **0 only** |
| AKOrN (attn) on CLEVR | 2 (seed 0 and its default 1234) |
| AKOrN (conv), ItrSA, Slot Attention | 1 each |

`PV2_0150` measured what a reseed is worth on a baseline: **FG-ARI spread 0.1339 on CLEVR**,
0.0535 on Tetrominoes, 0.0094 on Multi-dSprites. Our margins exceed all of those, which is the
most that two baseline seeds can buy us. It is not a substitute for our own.

Everything in PV2_0137–0151 is our seed 0 — on all three datasets, because all three are read
from the one CLEVR-trained core. Never report one checkpoint evaluated three times as a 3-seed
training mean, and never fill the generalisation column with the per-dataset 3-seed cores: those
answer a different question at a smaller budget.

## 5. Order of work

1. **Seeds 1 and 2 of the 253 core on CLEVR**, 10,000 × 10 at grid 64. Still the only item that
   converts every candidate in this project into a result. Needs a spending decision: ~18.5 h
   each on a rented 4090 (~$8/seed), or free on frontier but sequential.
2. **A second AKOrN (conv) seed.** 7 minutes, and it is the one thing that settles or unsettles
   our narrowest cell (§2.3).
3. **Score the two cont5000 branches** (Tetrominoes done and cached; Multi-dSprites running) on
   their own dataset **and on CLEVR** — does 0.8421 survive continued training elsewhere? This
   is the open question about whether the core can be improved without losing generality.
4. **Protocol B, labelled as what it is.** For AKOrN that is 300 epochs on our 10,000 (~3.5 h,
   "its recipe on our data budget") or on the full 70,000 (~24 h, "its recipe"). For Slot
   Attention it is the published 500k steps (`PV2_0139`, written, unrun). Pick per model and say
   which was run.
5. **Rotating Features and the complex autoencoders.** Each needs its own environment and a
   loader for our splits. Lowest priority: their published numbers already coincide with our
   protocol on Multi-dSprites, where grid 64 *is* the pixel protocol.

## 6. What the final report says, and in what order

1. **The headline**: our three metrics on three datasets, per-seed with mean and standard
   deviation, beside the **best-baseline-per-cell** composite, on the same image ids.
2. **Every baseline's own row**, with its published figure and its budget verdict (§2.4).
3. **The diagnostics table** (§3): predicted foreground against truth, own-metric agreement,
   background rule and the split that chose it.
4. **The narrow cell, named** (§2.3).
5. **The claim, scoped.** With seeds: "a CLEVR-trained core, frozen, beats the best of four
   baselines in every metric cell on three datasets at an identical data budget, two of which it
   never trained on." Without seeds, the same sentence with "at seed 0" in it, which is weaker
   and true.
6. **The config, split manifest version, evaluation contract version, code commit, reproduction
   commands, checkpoint identifiers, known limitations** — and the evidence that the score came
   from classifier masks on spikes, not from theta or PLV.
