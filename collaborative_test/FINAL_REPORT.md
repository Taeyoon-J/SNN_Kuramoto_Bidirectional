# Final report — what this project has established, and at what confidence

Written for the user and the collaborator. Every row states its seed count and, where one was
measured, its standard deviation. Nothing is presented as a 3-seed mean, because **no 3-seed
core mean exists**; where that matters the row says so.

Protocol for every row below unless noted: **patch contract, grid 64, 320 evaluation images,
`evaluate_patch_masks`, background id 0, ceiling 1.0, predictions never expanded to pixels.**
The scorer and its two verification values are in `evaluation_howto/`.

---

## 1. The headline — one core trained on CLEVR, reused

| dataset | configuration | patch_fg_ari | patch_foreground_iou | patch_matched_object_iou | seeds |
|---|---|---|---|---|---|
| **CLEVR** fixed320 | 253 core alone | 0.8421 | 0.7545 | 0.6410 | 1 |
| | **+ head, label-free recipe** | **0.8468** | **0.7574** | **0.6456** | 1 core, 5 head |
| | gain | **+0.0047 ± 0.0004** | **+0.0030 ± 0.0003** | **+0.0050 ± 0.0005** | head-seed std |
| | + head, rule B′ (uses validation labels) | 0.8507 | 0.7830 | 0.6709 | 1 |
| **Tetrominoes** test | 253 core alone, **never saw this data** | **0.8580** | **0.7885** | **0.8173** | 1 |
| | + head | — | — | — | **withdrawn**¹ |
| **Multi-dSprites** test | 253 core alone, **never saw this data** | **0.6974** | **0.8177** | **0.7628** | 1 |
| | **+ head, label-free recipe** | **0.7041** | **0.8213** | **0.7660** | 1 core, 5 head |
| | gain | **+0.0064 ± 0.0011** | **+0.0057 ± 0.0027** | **+0.0059 ± 0.0023** | head-seed std |

¹ The Tetrominoes head gain was reported as +0.0006 / +0.0010 / +0.0015 and is **withdrawn**
(`PV2_0146`): over five head seeds it is **−0.0020 / +0.0003 / −0.0008**, negative on two of
three metrics and inside the head-seed spread on all three. It was head seed 0's draw.

**The generalisation claim, stated exactly:** a core trained **only on CLEVR** and then frozen
scores **0.8580** FG-ARI on Tetrominoes and **0.6974** on Multi-dSprites, datasets it never saw.
The score comes from a classifier on spikes (`spike_synchrony_components` →
`foreground_induced_merge`), with no head required for those two numbers. A head adds a small,
reproducible gain on CLEVR and Multi-dSprites and nothing measurable on Tetrominoes.

---

## 2. The controlled test — is a CLEVR-trained core really better than a core trained on the target?

Grid 64, 10,000 × 10 on both sides, `min_group` 8 on both sides, the **same 320 image ids
verified equal**, same evaluation function. Only the training data and the random
initialisation differ.

| dataset | core trained on | fg_ari | fg_iou | obj_iou | regions/image | truth objects |
|---|---|---|---|---|---|---|
| **Tetrominoes** | **CLEVR** | **0.8580** | **0.7885** | **0.8173** | 22.1 | 3.0 |
| | itself | 0.5448 | 0.6217 | 0.4434 | 168.5 | |
| | itself + head | 0.5578 | 0.6995 | 0.6184 | | |
| | **CLEVR advantage** | **+0.3002** | +0.0890 | +0.1989 | | |
| **Multi-dSprites** | **CLEVR** | **0.6974** | **0.8177** | **0.7628** | 7.6 | 2.5 |
| | itself | 0.1201 | 0.0597 | 0.0050 | 413.6 | |
| | itself + head (B′, epoch 0) | 0.3269 | 0.2844 | 0.5219 | | |
| | **CLEVR advantage** (both with their own head) | **+0.3772** | +0.5369 | +0.2441 | | |

Best configuration against best configuration, each side carrying its own B′ head, the CLEVR
core still leads by **+0.3002** (Tetrominoes) and **+0.3772** (Multi-dSprites) FG-ARI.

**What the head repairs is a measure of how broken the core was.** On the fragmented
Multi-dSprites core it lifts matched-object IoU **104×** (0.0050 → 0.5219) and moves **23.8%** of
patches; on the CLEVR core the same structure moves 0.3–2% and gains hundredths. B′ selected
**epoch 0** there — the head before any update — so the repair comes from the architecture (a
background candidate plus locality-constrained reassignment, change A), not from learned
weights.

**Confidence.** All four rows are seed 0, so there is no error bar on them directly. The nearest
measured core-seed standard deviations are **0.0396** (Tetrominoes) and **0.0826**
(Multi-dSprites), from three finished seeds at grid 32 / 3,500 × 10 (`PV2_0144`). The gaps are
**8×** and **6×** those. Initialisation does not plausibly account for either.

**Why it happens.** Training on these datasets is not broken — on Tetrominoes it lifts FG-ARI
from **0.0100** (untrained core) to 0.5448, a factor of 54. It simply stops short. What tracks
performance is how far `positive_spike` descends, and all three cores below are grid 64 at
10,000 × 10, each scored on its own data:

| core trained on | `positive_spike` | score on its own data |
|---|---|---|
| CLEVR | **−83.2%** | **0.8421** |
| Tetrominoes | −33.3% | 0.5448 |
| Multi-dSprites | **−0.1%** | 0.1201 |

Strictly monotone. `phase_primary`, flat on both non-CLEVR datasets, predicts nothing. CLEVR
starts that term at 4.64 against 2.62 elsewhere — near where CLEVR *finishes* (2.46) — so CLEVR
is simply the dataset rich enough for this objective to learn grouping, and what it learns
carries.

---

## 3. Slot Attention — three senses, and we satisfy one

| sense | result |
|---|---|
| **1. Its published numbers** (Tetrominoes 99.5, Multi-dSprites 91.3, CLEVR**6** 98.8) | **we lose, and the comparison is invalid** — per-pixel scoring, decoder reconstruction, easier variants. `BENCHMARK_PLAN.md` already ruled it out. **Never claim this.** |
| **2. Its published checkpoint, our images, our protocol** | **we win: 0.8468 vs 0.3747** — but that checkpoint *tiles rather than segments* on our render (6.99 of 7 slots per frame, near-equal shares), so we are beating a failed transfer. Seed 0. |
| **3. Retrained on our render** | **not done.** `PV2_0139` holds the training loop; it needs a budget decision. **This is the only sense that supports "we beat Slot Attention."** |

Its background rule was not the problem: four rules fixed in advance and chosen on validation
all give the **identical** FG-ARI, because they only touch patches the truth calls background.

### Where we stand against the field whose protocol matches ours

For Multi-dSprites, `Dataset.labels` and `Dataset.pixels` are the **identical 64×64 array**
(verified), so our grid-64 number *is* the pixel number and comparison with the published
unsupervised synchrony family is legitimate:

| model (unsupervised, pixel ARI-FG) | Multi-dSprites |
|---|---|
| CAE | 0.05 |
| CAE++ | 0.51 |
| CtCAE | 0.56 |
| **ours, CLEVR-trained, zero-shot** | **0.6974** |
| SynCx | 0.82 |
| Rotating Features | 0.84 |

Fourth of six, where our records previously had this dataset as *"loses, at a grid where winning
was impossible."* **Ours is the only zero-shot row** — every other trained on Multi-dSprites. It
also matches the peer's **CLEVR mask-supervised** zero-shot numbers (0.6833 / 0.6979,
`PV2_0128`) without any supervision.

---

## 4. Why 252 lost to 253 on identical data

Identical budgets — both 6,250 updates, 100,000 images seen, 10,000 unique train images, 10
epochs. First 50 updates against last 50, from each run's own `training.jsonl`:

| | 252 | 253 |
|---|---|---|
| `phase_primary` | 7.2389 → 7.0511 (**−2.6%**) | 4.6355 → 2.4556 (**−47.0%**) |
| `positive_spike` | 15.6905 → 10.4814 (−33.2%) | 15.3876 → 2.5839 (**−83.2%**) |
| `gradient_norm` | 47.96 → 247.27 (**+416%**) | 6.08 → 6.27 (+3.2%) |
| core mean relative parameter change | **0.0700** | **0.1124** |
| gradient mass into the core | 3.408e+05 | 9.140e+04 |

252's core received **3.7× the gradient and moved 0.62× as far**; its 12-tensor encoder received
**65×** the gradient and moved *less*. More gradient producing less motion is cancellation.

**Supervision made it worse for a structural reason.** It enters by two routes and neither can
improve what is scored. The mask teacher is **detached** — `loss.py:65` is
`if model.variant==252: features=features.detach()` — so the labels train the head and cannot
reach the core; that term improved 52.9%, which is where the head's 177% parameter movement
went, and the head is not what the score is read from. The gamma teacher does reach the core and
went the **wrong way**: `gamma_anchor` ended **62× worse** (0.00016 → 0.01591).

The symptom is in `curve252.json`: `foreground_iou` is **0.2021225 at all five checkpoints,
bitwise identical** — not a plateau but a constant, because the readout designated no background
and the metric collapsed to the truth's foreground fraction.

---

## 5. What is missing, and what it would settle

| missing | what it would settle | cost |
|---|---|---|
| **CLEVR core seeds 1 and 2** | the only real error bars. Specifically: does the head gain survive a *different* core? The core comparison is already 6–8× its nearest measured std. | 18.5 h each on a rented RTX 4090, ~$8 per seed; two in parallel ≈ 1 day, ~$16. **51–64 h each on frontier — too slow.** |
| **Slot Attention retrained on `clevr_with_masks`** | sense 3 above — the only honest form of "we beat Slot Attention". | days per seed; `PV2_0139` holds the loop |

Until the first exists, every number in sections 1 and 2 is **seed 0**, and the head gains carry
only **head-seed** standard deviations, which do not cover core variance.

## 6. Reproduction

| | |
|---|---|
| core | `/work/USERS/tkim1/share/native64_core_253/{resume.pt, initial.pt}`, sha256 `e56b1a4219ab606b35952dab99ce0623…`, 6,250 updates, 10,000 × 10, grid 64, seed 0 |
| heads | `/work/USERS/tkim1/sw0255b/fit5/*.pt` — `recipe2_clevr.pt` is the label-free headline |
| scorer | `evaluation_howto/score_any.py`, with two values that confirm a correct setup |
| caches | `/work/USERS/tkim1/sw0255b/ev` (CLEVR core), `ev261` (target-trained cores) |
| **do not load** | `sw0255b/run/jobs/model253/clevr/seed0/resume.pt` — an aborted 8-update checkpoint whose **file size is byte-identical** to the real one |

`initial.pt` is required alongside `resume.pt`; the feature normalization lives there.
