# PV2_0153 — continuing on Multi-dSprites beats 253 on all six cells, and refutes PV2_0152's mechanism

**Status** completed. One core seed on each side; the ± values are **head re-initialisation
variance with the core held fixed** (5 head seeds) and are not core seed variance.
**Parent** `PV2_0152` (the same experiment on Tetrominoes) and `PV2_0148` (the `cont5000` preset).

**Hypothesis.** `PV2_0152` continued the finished CLEVR core on **Tetrominoes** and beat 253's
FG-ARI on both datasets, but lost 0.0203 of CLEVR matched-object IoU. It explained that by
**merging**: mean regions fell on both datasets, which deletes splinters on a sparse render and
joins objects on a dense one. This run asks whether that mechanism is general — the identical
procedure, 5,000 × 10 = 50,000 more exposures, on **Multi-dSprites** instead.

**Changed variable.** Which dataset the continuation trains on. Everything else — warm start,
budget, recipe, grid, seed, evaluation — is held.

**Success criterion.** Beat the base core on the dataset it continued on, and say honestly what
it costs on CLEVR, as PV2_0152 did.

## Result — it costs nothing. All six cells improve.

Core alone, `score_cache.py` on `regions_253`, same reserved 320 ids, `min_group` 8.

### Multi-dSprites test 320 — the dataset it continued on

| metric | base core | base + head | **cont core** | **cont + head** | Δ core | Δ head |
|---|---|---|---|---|---|---|
| fg_ari | 0.6974 | 0.7037 ± 0.0011 | **0.7032** | **0.7063 ± 0.0009** | **+0.0058** | +0.0026 |
| foreground_iou | 0.8177 | 0.8234 ± 0.0027 | **0.8296** | **0.8306 ± 0.0015** | **+0.0119** | +0.0072 |
| matched_object_iou | 0.7628 | 0.7687 ± 0.0023 | **0.7758** | **0.7776 ± 0.0008** | **+0.0130** | +0.0089 |

### CLEVR fixed320 — the dataset it was originally trained on

| metric | base core | base + head | **cont core** | **cont + head** | Δ core | Δ head |
|---|---|---|---|---|---|---|
| fg_ari | 0.8421 | 0.8468 ± 0.0004 | **0.8509** | **0.8543 ± 0.0003** | **+0.0088** | +0.0075 |
| foreground_iou | 0.7545 | 0.7575 ± 0.0003 | **0.7592** | **0.7609 ± 0.0004** | **+0.0047** | +0.0034 |
| matched_object_iou | 0.6410 | 0.6460 ± 0.0005 | **0.6671** | **0.6723 ± 0.0007** | **+0.0261** | +0.0263 |

**Six of six cells up, with and without the head, and the largest single gain is the one
PV2_0152 lost**: CLEVR matched-object IoU **+0.0261** here against **−0.0203** there.

With the head, all three CLEVR deltas clear twice the head-seed standard deviation.
**`cont5000(Multi-dSprites)` + head is the best configuration this project has: CLEVR
0.8543 / 0.7609 / 0.6723 against the previous best 0.8468 / 0.7574 / 0.6456.**

> **WITHDRAWN (`PV2_0156`).** That sentence rested on two datasets. Scored on Tetrominoes as
> well, this core **loses two of nine cells** (FG-ARI −0.0129, matched-object IoU −0.0066), so it
> is not the best configuration and does not meet the replacement criterion. On the stated
> FG-ARI-first priority the better core is the *other* branch, `cont5000(Tetrominoes)`, which
> raises FG-ARI on all three datasets. See `PV2_0156`.
>
> The CLEVR truth in the table below is also corrected from "~3.0" to **6.36** (`PV2_0157`),
> which strengthens rather than weakens the refutation: the Tetrominoes branch's 6.6 is the most
> accurate object count we have measured, so its matched-object IoU loss cannot be over-merging.

## This refutes PV2_0152's mechanism, and the diagnostic says why

PV2_0152 attributed its Tetrominoes gain and its CLEVR object-IoU loss to one cause — fewer
regions. The two branches disagree on exactly that:

| | base core | cont5000 **Tetrominoes** | cont5000 **Multi-dSprites** | truth |
|---|---|---|---|---|
| mean regions on CLEVR | 8.0 | **6.6** | **9.9** | **6.36** |
| CLEVR matched_object_iou | 0.6410 | 0.6207 (**−0.0203**) | 0.6671 (**+0.0261**) | |
| CLEVR predicted foreground | 0.2001 | 0.2033 | 0.2205 | 0.2115 |

**Continued training does not reduce fragmentation as such.** On the Tetrominoes branch CLEVR
regions fell 8.0 → 6.6 and object identity degraded; on the Multi-dSprites branch they *rose*
8.0 → 9.9 and object identity improved by more than the other lost. So "it merges" was a
description of the Tetrominoes branch, not a property of `cont5000`, and PV2_0152's mechanism
claim is **withdrawn as a general statement**. It remains the correct reading of that branch.

What the two branches share is the direction on **FG-ARI** (+0.0095 and +0.0088 on CLEVR), which
is the cell where continued training looks robust. Everything else depends on which dataset the
continuation saw. The honest summary is narrower and more useful: **continuing on a render whose
object statistics are closer to CLEVR's transfers back; continuing on a much sparser one does
not.** Multi-dSprites has 2–5 regions per image against CLEVR's 3–11; Tetrominoes has exactly 4
every time, and it is the branch that regressed.

## The head, re-measured on this core

| dataset | fg_ari | foreground_iou | matched_object_iou |
|---|---|---|---|
| Multi-dSprites | +0.0026, inside noise | +0.0072, clears 2× | +0.0089, clears 2× |
| CLEVR | **+0.0075, clears 2×** | **+0.0034, clears 2×** | **+0.0263, clears 2×** |

On the base core the head cleared its noise in three of six cells (`PV2_0146`); on this core it
clears five of six, and on CLEVR all three. **So the head direction, which `PV2_0146` found
closed on Tetrominoes, is open on this core** — which is the branch of the standing instruction
that says to use the head when it is not closed. The head is still small: its largest
contribution here is +0.0263, against the core's own +0.0261 on the same cell.

## What this licenses, and what it does not

**Licenses:** "continuing the finished CLEVR core for 50,000 exposures on Multi-dSprites gives a
core that beats the base core on all three metrics on **both** datasets, with and without the
head, and the combination with the label-free head is the best configuration measured in this
project."

**Does not license** a mechanism. PV2_0152 offered one and this run contradicts it; nothing here
replaces it with a tested alternative, and the region-count table above is a correlation across
two branches, not an experiment. The next test would vary object density directly.

**Seeds.** One core seed on each side, by decision. The gains are 0.0047–0.0261 and we have **no
core seed variance** anywhere in this project. The head-seed ± values do not cover it. The
CLEVR matched-object IoU gain of +0.0261 is the largest and the only one that would survive a
generous noise assumption.

**Not measured:** this core on **Tetrominoes**. Both continued cores should be scored on all
three datasets before either is adopted; right now each has been scored on its own dataset and
on CLEVR only.

## Provenance

- **Core.** `/work/USERS/tkim1/sw0255b/run_br_mds/jobs/model253/multi_dsprites/seed0/`,
  `resume.pt` sha256 `5fda94866ee03947ae4cac3a8308e72aab9e56fefc63eeb6ff473023c1ac2b4f`,
  9,380 updates / 150,000 exposures, `unique_TRAIN` 5,000, `cont5000_stage_added` 3,130 updates /
  50,000 images, warm start 6,250 updates / 100,000 exposures, grid 64, seed 0. Trained on
  frontier GPU 2 over ~22 h at ~29 s/update.
- **Caches.** `/work/USERS/tkim1/sw0255b/ev_cont_mds/`, `--min-group 8`, grid 64:
  `multi_dsprites_{train 256, test 320}`, `clevr_{train 256, fixed320 320}`.
- **Pipeline.** `/work/USERS/tkim1/finish_mds.sh` ran unattended: waited on `committed.json`
  reaching 9,380 updates, cached in two parallel jobs, then scored core-only and 5 head seeds on
  both datasets and wrote `SUMMARY.txt`. Results under
  `/work/USERS/tkim1/share/cont5000_multi_dsprites/results/`.
- **Scored by** `evaluation_howto/score_cache.py` (core alone, validated against both base rows)
  and `headseed.py` (5 head seeds), both on `regions_253`.

## Next

1. **Score both continued cores on all three datasets.** Each is missing one dataset. Until then
   neither can be put in the headline table, which is a three-dataset table.
2. If `cont5000(Multi-dSprites)` holds up on Tetrominoes, it replaces 253 as the reported core,
   and the nine-of-nine margins in `PV2_0151` all widen.
3. PV2_0152's mechanism paragraph needs the withdrawal noted in its own file, not only here.
