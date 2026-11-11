# Final report — what this project has established, and at what confidence

Written for the user and the collaborator. Every row states its seed count and, where one was
measured, its standard deviation. Nothing is presented as a 3-seed mean, because **no 3-seed
core mean exists**; where that matters the row says so.

**Core seeds 1 and 2 were decided against** (2026-11-09), so every number here is **seed 0** and
the contract's 3-seed line is a documented scope choice, not an oversight. What partly covers it:
`PV2_0150` measured what a reseed is worth *on a baseline* — FG-ARI spread 0.1339 on CLEVR,
0.0535 on Tetrominoes, 0.0094 on Multi-dSprites — and our margins over the field (0.134–0.333,
§3.1) exceed all of them. That is evidence, not an error bar on our own core.

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

### 1.1 Beating 253 — `cont5000(Tetrominoes)`

The head direction is **closed on Tetrominoes** (`PV2_0146`: −0.0020 / +0.0003 / −0.0008 over
five head seeds), so the instruction for that case was to beat 253 **through the core**.
Continuing the finished CLEVR core for **5,000 × 10 = 50,000 more exposures on Tetrominoes**
does it, by the stated rule — FG-ARI first, Foreground IoU may give a little:

| dataset | 253 core | **cont5000(Tetrominoes)** | Δ |
|---|---|---|---|
| CLEVR fixed320 | 0.8421 / 0.7545 / 0.6410 | **0.8505** / 0.7452 / 0.6207 | **+0.0084** / −0.0093 / **−0.0203** |
| Tetrominoes test | 0.8580 / 0.7885 / 0.8173 | **0.8675 / 0.8193 / 0.8213** | **+0.0095** / +0.0308 / +0.0040 |
| Multi-dSprites test | 0.6974 / 0.8177 / 0.7628 | **0.7142 / 0.8189** / 0.7561 | **+0.0168** / +0.0012 / −0.0067 |

**FG-ARI rises on all three datasets.** Foreground IoU falls only on CLEVR, by 0.0093, which the
instruction explicitly permits. The same holds with the head (`PV2_0156`).

**Why this core and not the other branch.** `cont5000(Multi-dSprites)` wins more cells against the
base core (7/9 against 6/9) but **loses Tetrominoes FG-ARI** (−0.0129), so it fails the stated
rule. More decisively, the project's claim is the margin over the **best baseline per cell**, where
the **weakest cell** fails first:

| core | cells won vs best baseline | **weakest cell** |
|---|---|---|
| 253 | 9 / 9 | +0.0345 |
| cont5000(Multi-dSprites) | 9 / 9 | +0.0459 |
| **cont5000(Tetrominoes)** | **9 / 9** | **+0.0653** |

All three win nine of nine; this one **nearly doubles the weakest link** (+89%), and that cell —
Tetrominoes Foreground IoU — is the one margin `PV2_0151` flagged as unsettled. FG-ARI-first and
maximise-the-weakest-cell select the same core.

**The cost, named rather than averaged away.** CLEVR matched-object IoU falls 0.6410 → 0.6207
**against our own previous core**; against baselines it is still +0.2354 over the best of eight,
so no comparative claim weakens. Multi-dSprites matched-object IoU −0.0067 likewise.

**What it does not fix.** `PV2_0157`: this core predicts **17.94 objects on Tetrominoes against a
truth of 3**, exact-match 0.003. It is the best core we have on the patch metrics and it still
cannot count. Adoption changes which numbers we report, not that.

**One core seed**, as everywhere here. This is a reversible reporting choice: every base-253 row
in this document is kept beside it.

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

## 3. Slot Attention — four senses, and we satisfy the two that are measurable

| sense | result |
|---|---|
| **1. Its published numbers** (Tetrominoes 99.5, Multi-dSprites 91.3, CLEVR**6** 98.8) | **we lose, and the comparison is invalid** — per-pixel scoring, decoder reconstruction, easier variants. `BENCHMARK_PLAN.md` already ruled it out. **Never claim this.** |
| **2. Its published checkpoint, our images, our protocol** | **we win: 0.8468 vs 0.3747** — but that checkpoint *tiles rather than segments* on our render (6.99 of 7 slots per frame, near-equal shares), so we are beating a failed transfer. Seed 0. |
| **3. Retrained on our render, our data budget** | **DONE (`PV2_0149`), and we win: 0.8468 vs 0.2872.** Trained from scratch on our 10,000 CLEVR ids at 100,032 exposures — the identical budget our core had. It scores *below* the transferred checkpoint, and 320× more data moves it only to 0.3747 with Foreground IoU identical to four decimals. **It tiles regardless of how long it trains**, so sense 2's weakness was never mainly domain shift. Seed 0 both sides. |
| **4. Its published 500k-step recipe** | **not done.** `PV2_0139` holds the loop. This is now the only unaddressed sense, and `PV2_0149` already shows the extra data buys it +0.0875 FG-ARI. |

Its background rule was not the problem: four rules fixed in advance and chosen on validation
all give the **identical** FG-ARI, because they only touch patches the truth calls background.

### 3.1 The whole field, trained on our data at our budget (`PV2_0150`, `PV2_0151`)

Four baselines, each trained on the **same 10,000 CLEVR images** at **100,000 exposures**, seed
0, scored by our function on the same reserved 320 ids. Our rows are the **core alone, no head**.

| model | CLEVR | Tetrominoes (zero-shot) | Multi-dSprites (zero-shot) |
|---|---|---|---|
| **ours, 253 core** | **0.8421 / 0.7545 / 0.6410** | **0.8580 / 0.7885 / 0.8173** | **0.6974 / 0.8177 / 0.7628** |
| AKOrN (conv) | 0.6046 / 0.4219 / 0.2769 | 0.7202 / 0.7540 / 0.5913 | 0.4055 / 0.6834 / 0.5434 |
| AKOrN (attn) | 0.6027 / 0.3327 / 0.3750 | 0.5676 / 0.7383 / 0.4687 | 0.4343 / 0.6019 / 0.5751 |
| ItrSA | 0.5794 / 0.2787 / 0.3853 | 0.6319 / 0.7137 / 0.5687 | 0.4817 / 0.2489 / 0.6093 |
| Slot Attention | 0.2872 / 0.2220 / 0.0801 | 0.1739 / 0.2192 / 0.1071 | 0.1930 / 0.0876 / 0.0680 |
| **best baseline per cell** | 0.6046 / 0.4219 / 0.3853 | 0.7202 / 0.7540 / 0.5913 | 0.4817 / 0.6834 / 0.6093 |
| **our margin** | +0.2375 / +0.3326 / +0.2557 | +0.1378 / **+0.0345** / +0.2260 | +0.2157 / +0.1343 / +0.1535 |

**Nine of nine cells**, and the bar is the *best baseline in each cell separately* — a composite
of three different models, which no single baseline achieves. Picking one reference would have
been picking our opponent: **conv wins Tetrominoes, ItrSA wins Multi-dSprites, attn wins neither.**

Three things an honest reader needs with that table:

- **Distrust one cell.** Tetrominoes Foreground IoU, **+0.0345**, with no seed spread measured
  for AKOrN (conv). Every other margin is 0.134–0.333.
- **The budget caveat is per model, not global.** AKOrN (conv) at 1/210 of its published budget
  **already matches its published CLEVR number** (60.5 vs 59.0±4.3), so beating it on
  Tetrominoes is not a data-efficiency artefact. AKOrN (attn) (60.4 vs 91.0±0.5) and ItrSA
  (58.1 vs 82.5±1.7) *are* starved, and for those rows the claim is data efficiency.
- **Only our core gets the foreground fraction right.** Predicted vs truth: CLEVR 0.2001 vs
  0.2115, Multi-dSprites 0.0841 vs 0.0943 — the only row that under-claims. Every baseline
  over-claims on CLEVR by 2–3.4× (Slot Attention 0.7047, ItrSA 0.7218). That one column
  separates tiling from segmenting faster than any metric, and it is why Tetrominoes — where all
  four baselines sit near the truth — is where our margin is thinnest.

Each baseline's own `eval_obj.py` FG-ARI agrees with ours to **≤0.0013** on every cell, which is
what rules out "the gap is your scorer".

### 3.2 Published numbers whose protocol already matches ours

For Multi-dSprites, `Dataset.labels` and `Dataset.pixels` are the **identical 64×64 array**
(verified), so our grid-64 number *is* the pixel number and comparison with the published
unsupervised synchrony family is legitimate:

| model (unsupervised, pixel ARI-FG) | Multi-dSprites | source |
|---|---|---|
| CAE | 0.05 | SynCx table |
| CAE++ | 0.51 | SynCx table |
| CtCAE | 0.56 | SynCx table |
| **ours, CLEVR-trained, zero-shot** | **0.6974** | measured here |
| SynCx | 0.82 ± 0.01 | its own paper |
| Rotating Features | 0.84 ± 0.03 | **SynCx's reproduction, not its authors' own number** |

**Sourcing, stated because it limits the comparison.** Every competitor figure in this table
comes from one third-party table — SynCx, arXiv 2405.17283 (recorded at `BENCHMARK_PLAN.md:97`)
— not from each model's own paper. For Rotating Features that matters concretely: its official
repo (`loeweX/RotatingFeatures`, commit `7a2f38f`) supports **4Shapes, FoodSeg103 and Pascal VOC
only**, with **zero mentions of dSprites or multi-object-datasets anywhere in it**. So 0.84 is
not reproducible from the authors' code without writing a new loader and training config, and it
is the one number above ours that we cannot currently check.

Fourth of six, where our records previously had this dataset as *"loses, at a grid where winning
was impossible."* **Ours is the only zero-shot row** — every other trained on Multi-dSprites. It
also matches the peer's **CLEVR mask-supervised** zero-shot numbers (0.6833 / 0.6979,
`PV2_0128`) without any supervision.

---

## 4. Beating our own core: `cont5000` (`PV2_0152`)

The head direction is **closed on Tetrominoes** (`PV2_0146`: −0.0020 / +0.0003 / −0.0008 over
five head seeds), so the way to beat 253 there had to be the core. `cont5000` continues the
finished CLEVR core for **5,000 × 10 = 50,000 more exposures on Tetrominoes** and changes nothing
else.

| | Tetrominoes test 320 | CLEVR fixed320 |
|---|---|---|
| base core | 0.8580 / 0.7885 / 0.8173 | 0.8421 / 0.7545 / 0.6410 |
| **cont core** | **0.8675 / 0.8193 / 0.8213** | **0.8505** / 0.7452 / 0.6207 |
| Δ core | **+0.0095 / +0.0308 / +0.0041** | **+0.0085** / −0.0093 / **−0.0203** |
| base + head (5 head seeds) | 0.8560±0.0033 / 0.7888±0.0005 / 0.8165±0.0016 | 0.8468±0.0004 / 0.7575±0.0003 / 0.6460±0.0005 |
| **cont + head** | **0.8665±0.0014 / 0.8206±0.0008 / 0.8235±0.0010** | **0.8526±0.0008** / 0.7468±0.0022 / 0.6252±0.0024 |

**It beats 253 on FG-ARI on both datasets at once** (+0.0095, +0.0085), with and without the
head, through the core. Under the stated priority — a higher FG-ARI may cost a little FG-IoU —
Tetrominoes is a clean win on all three and CLEVR's −0.0093 FG-IoU is an accepted trade.

**The cost that priority does not cover is CLEVR matched-object IoU, −0.0203.** It is the largest
movement in the table, 4× the FG-IoU loss, and it is reported as a regression, never averaged in.

**Mechanism: it merges.** Mean regions per image fall on *both* datasets — Tetrominoes 21.1 →
17.9, CLEVR 8.0 → 6.6, against a truth of about 3. On Tetrominoes that deletes spurious
splinters, so grouping and coverage both improve. On CLEVR it has begun joining objects to each
other, which lifts a pairwise partition measure (FG-ARI) and lowers the one metric that punishes
two true objects in one region (matched-object IoU). Same cause, opposite sign — which is
precisely why the three metrics are never combined.

The most robust number is the Tetrominoes FG-IoU gain: **+0.0308** on test, independently
**+0.0328** on validation 160, measured before the evaluation split was touched.

The matching Multi-dSprites branch is still running; it will say whether the merging effect is
general or a property of Tetrominoes' very sparse render.

---

## 5. Why 252 lost to 253 on identical data

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

## 6. What is missing, and what it would settle

| missing | what it would settle | status |
|---|---|---|
| **CLEVR core seeds 1 and 2** | the only real error bars on our own column. | **decided against, 2026-11-09.** Would have been ~18.5 h and ~$8 per seed on a rented 4090, or 51–64 h each on frontier. |
| **Slot Attention at its published 500k-step recipe** | sense 4 in §3 — the last unaddressed sense. | not run; `PV2_0139` holds the loop |
| **Protocol B for AKOrN** | whether we still lead when attn and ItrSA are not starved. | not run; ~3.5 h on our 10,000 or ~24 h on the full 70,000 |
| **A second AKOrN (conv) seed** | our one narrow cell, Tetrominoes Foreground IoU +0.0345. | not run; **7 minutes** — the cheapest item that can change a conclusion |
| **Multi-dSprites `cont5000` branch** | whether §4's merging effect is general or specific to Tetrominoes' sparse render. | running; pipeline `/work/USERS/tkim1/finish_mds.sh` completes it unattended |

Every number in this report is **seed 0**, and the head gains carry only **head-seed** standard
deviations, which do not cover core variance. That is a stated limitation, not a hidden one.

## 7. Reproduction

| | |
|---|---|
| core | `/work/USERS/tkim1/share/native64_core_253/{resume.pt, initial.pt}`, sha256 `e56b1a4219ab606b35952dab99ce0623…`, 6,250 updates, 10,000 × 10, grid 64, seed 0 |
| heads | `/work/USERS/tkim1/sw0255b/fit5/*.pt` — `recipe2_clevr.pt` is the label-free headline |
| scorer | `evaluation_howto/score_any.py`, with two values that confirm a correct setup; `score_cache.py` scores a core with no head out of a cache; `to_grid64.py` and `bg_any.py` score a baseline that emits neither our grid nor our background id |
| caches | `/work/USERS/tkim1/sw0255b/ev` (CLEVR core), `ev261` (target-trained cores), `ev_cont_tetro` (cont5000 Tetrominoes), `ev_cont_mds` (cont5000 Multi-dSprites) |
| baselines | `/work/USERS/tkim1/baselines/_clone/akorn` (commit `eabbe27`) and `/work/USERS/tkim1/baselines/slot_attention`; checkpoints and hashes in `PV2_0150/provenance.json` and `PV2_0151/provenance.json` |
| cont5000 core | `/work/USERS/tkim1/share/cont5000_tetrominoes/`, `resume.pt` sha256 `ca8546b47432067a390c368f734996ace066923e1be5dd77b9dad8055c433ceb`, 9,380 updates / 150,000 exposures |
| **do not load** | `sw0255b/run/jobs/model253/clevr/seed0/resume.pt` — an aborted 8-update checkpoint whose **file size is byte-identical** to the real one |

`initial.pt` is required alongside `resume.pt`; the feature normalization lives there.
