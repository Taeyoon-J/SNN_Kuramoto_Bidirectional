# PV2_0152 — continued training beats 253 on FG-ARI on *both* datasets, and pays for it in object IoU

**Status** completed. Core is one seed on each side; the ± values are **head re-initialisation
variance with the core held fixed** (5 head seeds) and are not core seed variance.
**Parent** `PV2_0148` (the `cont5000` preset) and `PV2_0137` / `PV2_0146` (the base rows).

**Hypothesis.** The head direction is closed on Tetrominoes (`PV2_0146`: the label-free head's
gain there is −0.0020 / +0.0003 / −0.0008 over 5 head seeds, inside noise). The standing
instruction for that case is to leave the head alone and **find a way to beat 253 through the
core**. `cont5000` is that attempt: take the finished CLEVR core and continue it for
**5,000 × 10 = 50,000 more exposures on Tetrominoes**, changing nothing else.

**Changed variable.** 3,130 additional optimizer updates on Tetrominoes images, warm-started
from the CLEVR core at 6,250 updates / 100,000 exposures. Same recipe, same grid, same seed.

**Success criterion.** Beat the base 253 core on Tetrominoes, and say honestly what it costs on
CLEVR. Under the standing instruction a higher FG-ARI may be bought with a slightly lower
FG-IoU.

## Result

Both sides scored by the identical code path (`score_cache.py` on `regions_253`, validated by
reproducing both base rows exactly). Head rows are the fixed label-free recipe
(`local_rgb_trust`, 2 epochs, no labels from the target dataset), 5 head seeds, mean ± std.

### Tetrominoes test 320 — the dataset it continued on

| metric | base core | base + head | **cont core** | **cont + head** | Δ core | Δ head |
|---|---|---|---|---|---|---|
| fg_ari | 0.8580 | 0.8560 ± 0.0033 | **0.8675** | **0.8665 ± 0.0014** | **+0.0095** | +0.0105 |
| foreground_iou | 0.7885 | 0.7888 ± 0.0005 | **0.8193** | **0.8206 ± 0.0008** | **+0.0308** | +0.0318 |
| matched_object_iou | 0.8173 | 0.8165 ± 0.0016 | **0.8213** | **0.8235 ± 0.0010** | **+0.0041** | +0.0070 |

All three up, and the FG-IoU gain was reproduced independently on validation 160
(+0.0328 there against +0.0308 here) before the evaluation split was touched.

### CLEVR fixed320 — the dataset it was originally trained on

| metric | base core | base + head | **cont core** | **cont + head** | Δ core | Δ head |
|---|---|---|---|---|---|---|
| fg_ari | 0.8421 | 0.8468 ± 0.0004 | **0.8505** | **0.8526 ± 0.0008** | **+0.0085** | +0.0058 |
| foreground_iou | 0.7545 | 0.7575 ± 0.0003 | 0.7452 | 0.7468 ± 0.0022 | **−0.0093** | −0.0107 |
| matched_object_iou | 0.6410 | 0.6460 ± 0.0005 | 0.6207 | 0.6252 ± 0.0024 | **−0.0203** | −0.0208 |

**This is the surprise.** Training on Tetrominoes for 50,000 exposures **raised CLEVR FG-ARI by
+0.0085** — 21× the base head's own std on that metric — while *lowering* CLEVR Foreground IoU
by 0.0093 and matched-object IoU by 0.0203.

### The headline both tables support

**`cont5000` beats the base 253 core on FG-ARI on both datasets at once** — +0.0095 on
Tetrominoes and +0.0085 on CLEVR — which is what the standing instruction asked for, and it does
it through the core, with the head direction closed. Under the stated priority (a higher FG-ARI
may cost a little FG-IoU) the Tetrominoes column is a clean win and the CLEVR column is an
accepted trade on FG-IoU.

**The part that is not covered by that priority is CLEVR matched-object IoU, −0.0203.** That is
the largest single movement in either table and it is a loss. It is not an FG-IoU-style "a
little lower"; it is 4× the CLEVR FG-IoU loss and 41× the base head's std on that metric.

## Why: it merges, and merging helps one metric and hurts another

> **The general form of this section is WITHDRAWN (`PV2_0153`).** The same procedure on
> Multi-dSprites *raises* CLEVR's region count (8.0 → 9.9, against 6.6 here) and *raises* CLEVR
> matched-object IoU by +0.0261, where this branch lost 0.0203. So continued training does not
> reduce fragmentation as such; it did on this branch. Everything below is the correct reading
> of the Tetrominoes branch and nothing more.

Mean regions per image, against a truth of about 3:

| dataset | base core | cont core | truth |
|---|---|---|---|
| Tetrominoes | 21.1 | **17.9** | ~3 |
| CLEVR | 8.0 | **6.6** | ~3 |

Continued training **reduces fragmentation on both datasets**, including the one it never saw
more of. Predicted foreground moves with it: Tetrominoes 0.2842 → 0.2661 against a truth of
0.2368 (over-claim shrinks, FG-IoU rises +0.0308), CLEVR 0.2001 → 0.2033 against 0.2115
(slightly *closer* to the truth, yet FG-IoU falls).

That pair of facts is the mechanism. **On Tetrominoes the merging removes spurious splinters**,
so grouping and coverage both improve. **On CLEVR the merging has started to join objects to
each other**: FG-ARI rewards getting the within/between-object partition right at the pair level
and improves, while matched-object IoU — which matches each true object to one predicted region
— is exactly the metric that punishes two objects landing in one region. Fewer regions than
before (6.6, against a truth of 3.0) on a render with more and smaller objects is how you buy
FG-ARI with object identity.

So the CLEVR loss is not noise and not a scoring artefact; it is the same merging that produced
the Tetrominoes gain, applied to a dataset where it has gone one step too far.

## The head, re-measured on the new core

`PV2_0146`'s finding survives and sharpens. Per metric, delta against head-seed std:

| dataset | fg_ari | foreground_iou | matched_object_iou |
|---|---|---|---|
| Tetrominoes | −0.0011, inside noise | +0.0013, inside noise | **+0.0022, clears 2×** |
| CLEVR | **+0.0021, clears 2×** | +0.0016, inside noise | +0.0044, inside noise |

The head is still worth a little and still cannot be the story: on the new core it clears its own
noise in **one metric per dataset**, and on Tetrominoes it still fails to help FG-ARI. Every
conclusion above is carried by the core.

## What this licenses, and what it does not

**Licenses:** "continuing a finished CLEVR core for 50,000 exposures on Tetrominoes beats the
base core's FG-ARI on both Tetrominoes (+0.0095) and CLEVR (+0.0085), with and without the head,
by reducing fragmentation — and costs 0.0203 of CLEVR matched-object IoU."

**Does not license** "cont5000 is better." One core seed on each side. The gains are 0.008–0.031
and we have **no core seed variance at all** — `PV2_0141` measured a 0.0453 arm-selection spread
on Tetrominoes FG-ARI, which is larger than the +0.0095 FG-ARI gain here, though it is a
different kind of noise (head arm/epoch, not core init). The FG-IoU gain of +0.0308, reproduced
at +0.0328 on a separate split, is the most robust number in this test.

**Do not read it as "training elsewhere is free for CLEVR."** It is not: object identity on
CLEVR degrades measurably.

## Provenance

- **Core.** `/work/USERS/tkim1/share/cont5000_tetrominoes/{resume.pt,initial.pt}`,
  `resume.pt` sha256 `ca8546b47432067a390c368f734996ace066923e1be5dd77b9dad8055c433ceb`,
  9,380 updates / 150,000 exposures, `unique_TRAIN 5,000`, `cont5000_stage_added` 3,130 updates /
  50,000 images, warm start `10,000×10 = 100,000 exposures, 6,250 updates`, grid 64, seed 0.
- **Where it was trained.** A rented RTX 4090. The run crashed *after* training, in
  `engine.py:229`'s post-training `evaluate()` → `preflight`, because
  `targets_tetrominoes_grid32.pt` was never shipped to that box. Training was unaffected; the
  checkpoint was pulled to frontier and the hash verified end to end.
- **Caches.** `/work/USERS/tkim1/sw0255b/ev_cont_tetro/`, `--min-group 8`, grid 64:
  `tetrominoes_{train 256, validation 160, test 320}`, `clevr_{train 256, validation 320,
  fixed320 320}`. ~10–11 s/frame.
- **Scored by** `evaluation_howto/score_cache.py` (core alone) and `headseed.py` (5 head seeds),
  both on `regions_253`.
- **Base rows** from `PV2_0137` (core) and `fit5/headseed_{clevr,tetrominoes}.json` (head).

## Next

1. **Done (`PV2_0153`): the merging effect is NOT general.** The Multi-dSprites branch improves
   all six cells, including the CLEVR matched-object IoU this branch lost. Of the two branches,
   that one is the better core and this one is the cautionary case.
2. **If `cont5000` is adopted, CLEVR matched-object IoU has to be reported as a regression**, not
   folded into an average. The three metrics are never combined, which is exactly why this test
   is readable at all.
3. Core seeds remain unrun **by decision**: everything in this project is seed 0 and is reported
   as such.

## Reproduce

```bash
# the four rows, from the cached activities
C=/work/USERS/tkim1/sw0255b/benchmark_codes; E=/work/USERS/tkim1/sw0255b/ev_cont_tetro
python $C/score_cache.py --cache $E/tetrominoes_test  --dataset tetrominoes --split test
python $C/score_cache.py --cache $E/clevr_fixed320    --dataset clevr       --split fixed320
python $C/headseed.py --dataset tetrominoes --train-cache $E/tetrominoes_train \
  --eval-cache $E/tetrominoes_test --eval-split test --arm local_rgb_trust --epochs 2 --seeds 5
python $C/headseed.py --dataset clevr --train-cache $E/clevr_train \
  --eval-cache $E/clevr_fixed320 --eval-split fixed320 --arm local_rgb_trust --epochs 2 --seeds 5
# (each also needs --code --config --run --out; see /work/USERS/tkim1/finish_261.sh)
```
