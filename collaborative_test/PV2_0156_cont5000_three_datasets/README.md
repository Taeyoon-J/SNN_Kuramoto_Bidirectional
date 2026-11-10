# PV2_0156 — both continued cores, on all three datasets: neither replaces 253, and the better one is the opposite of what PV2_0153 said

**Status** completed. One core seed throughout, by decision; ± values are head re-initialisation
variance with the core held fixed.
**Parent** `PV2_0152` (Tetrominoes branch) and `PV2_0153` (Multi-dSprites branch).

**Why this test.** Each branch had been scored on **its own dataset and CLEVR only**, so neither
could enter a three-dataset table. This fills the two missing cells:
`cont5000(Multi-dSprites)` on Tetrominoes, and `cont5000(Tetrominoes)` on Multi-dSprites.

**Success criterion, fixed before looking.** A core replaces 253 only if it wins **every cell it
is scored on**.

## Result — neither core wins everywhere

Core alone, `score_cache.py` on `regions_253`, same reserved 320 ids per dataset, `min_group` 8.
Deltas are against the base 253 core.

### cont5000(Multi-dSprites) — 7 of 9

| dataset | base core | cont core | Δ |
|---|---|---|---|
| CLEVR | 0.8421 / 0.7545 / 0.6410 | **0.8509 / 0.7592 / 0.6671** | +0.0088 / +0.0047 / +0.0261 |
| **Tetrominoes** | 0.8580 / 0.7885 / 0.8173 | 0.8451 / **0.7999** / 0.8107 | **−0.0129** / +0.0114 / **−0.0066** |
| Multi-dSprites | 0.6974 / 0.8177 / 0.7628 | **0.7032 / 0.8296 / 0.7758** | +0.0058 / +0.0119 / +0.0130 |

### cont5000(Tetrominoes) — 6 of 9

| dataset | base core | cont core | Δ |
|---|---|---|---|
| **CLEVR** | 0.8421 / 0.7545 / 0.6410 | **0.8505** / 0.7452 / 0.6207 | +0.0084 / **−0.0093** / **−0.0203** |
| Tetrominoes | 0.8580 / 0.7885 / 0.8173 | **0.8675 / 0.8193 / 0.8213** | +0.0095 / +0.0308 / +0.0040 |
| Multi-dSprites | 0.6974 / 0.8177 / 0.7628 | **0.7142 / 0.8189** / 0.7561 | **+0.0168** / +0.0012 / **−0.0067** |

With the head the picture is the same in sign on every cell; the head rows are in
`CROSS_SUMMARY.txt`.

## Two corrections to `PV2_0153`

**1. "The best configuration this project has" is withdrawn.** `PV2_0153` called
`cont5000(Multi-dSprites)` + head the best configuration measured here, on the strength of six
cells. With Tetrominoes measured it **loses two of nine** — FG-ARI −0.0129 and matched-object IoU
−0.0066 — so it does not meet the replacement criterion.

**2. The better branch is the other one, under the stated priority.** The standing instruction is
that a higher FG-ARI may be bought with a slightly lower Foreground IoU. On FG-ARI alone:

| | CLEVR | Tetrominoes | Multi-dSprites |
|---|---|---|---|
| cont5000(Multi-dSprites) | +0.0088 | **−0.0129** | +0.0058 |
| **cont5000(Tetrominoes)** | **+0.0084** | **+0.0095** | **+0.0168** |

**`cont5000(Tetrominoes)` raises FG-ARI on all three datasets.** Its costs are CLEVR Foreground
IoU −0.0093 — small, and explicitly allowed — and matched-object IoU −0.0203 on CLEVR and
−0.0067 on Multi-dSprites, which is not covered by that allowance and stays a reported
regression. `PV2_0152` treated that core as already disqualified and this record withdraws that
too: on the stated priority it is the adoption candidate, with one named regression.

## The result neither branch predicted

**A core continued on Tetrominoes improves Multi-dSprites more than a core continued on
Multi-dSprites does**: FG-ARI **+0.0168** against **+0.0058**. It is the largest FG-ARI gain
anywhere in this test, and it is cross-dataset — the continuation data and the improved dataset
are different.

That is evidence against reading `cont5000` as domain adaptation. If continuing on a dataset
mainly tuned the core to that dataset, the diagonal would be strongest; it is not. Tetrominoes is
the sparsest render (exactly 3 objects every image, against Multi-dSprites' 1–4 and CLEVR's
2–10), and continuing there helps the other two renders' grouping while costing object identity
on the densest one. No mechanism is claimed; `PV2_0157` measures the object counts that would
have to be part of one.

## What this licenses

**Licenses:** "50,000 more exposures on a second dataset raises FG-ARI on all three datasets when
that dataset is Tetrominoes, at a cost of 0.0203 CLEVR matched-object IoU; and neither continued
core beats the base core on all nine cells."

**Does not license** replacing 253 in the headline table. Under the fixed criterion — win every
cell scored — both fail. Under the FG-ARI-first priority, `cont5000(Tetrominoes)` qualifies with
a named exception, which is a decision to take explicitly rather than a result to assume.

**Seeds.** One core seed. The gains run 0.0012–0.0308 and no core seed variance exists anywhere
in this project; the head-seed ± values do not cover it. The largest movements — Tetrominoes
Foreground IoU +0.0308 and CLEVR matched-object IoU −0.0203 — are the two that would survive a
generous noise assumption, and they point in opposite directions.

## Provenance

| | |
|---|---|
| cores | `run_br_mds/jobs/model253/multi_dsprites/seed0/` sha256 `5fda9486…`; `share/cont5000_tetrominoes/` sha256 `ca8546b4…`. Both 9,380 updates / 150,000 exposures, grid 64, seed 0 |
| new caches | `ev_cont_mds/tetrominoes_{train 256, test 320}`, `ev_cont_tetro/multi_dsprites_{train 256, test 320}`, `--min-group 8` |
| pipeline | `/work/USERS/tkim1/finish_cross.sh`, unattended: two caches in parallel, then core-only and 5 head seeds per new cell, then the table |
| scored by | `evaluation_howto/score_cache.py` and `headseed.py`, both on `regions_253` |
| full table | `CROSS_SUMMARY.txt` in this directory |

## Next

1. **Decide explicitly** whether FG-ARI-first promotes `cont5000(Tetrominoes)` over 253, given
   the CLEVR matched-object IoU regression. That is a judgement about the goal, not a measurement.
2. If it is promoted, `PV2_0151`'s nine-of-nine margins are recomputed against it — six of the
   nine baseline cells would widen and three (CLEVR FG-IoU, CLEVR obj-IoU, Multi-dSprites obj-IoU)
   would narrow.
3. `PV2_0155` (the 1,000-image scaling curve) is running and asks whether any of this converges.
