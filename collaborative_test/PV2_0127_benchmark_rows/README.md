# PV2_0127 — the CLEVR benchmark row at three seeds, and a Tetrominoes seed that died

- **status** completed for CLEVR; Tetrominoes partially, with one arm retraining
- **parent** `PV2_0125` (the pixel protocol), `PV2_0121` (Tier 1 training)
- **branch** `patch_v2`
- **contract** `v2-pixel` for the benchmark datasets, contract 2 for CLEVR; metrics from
  `collaborative_test/evaluation.py` unchanged
- **protocol decision, by the user:** the grid is chosen **per dataset from its measured
  ceiling** rather than unified. CLEVR 32, Multi-dSprites 64, Tetrominoes 32. The table
  states each row's grid and its ceiling.

## Why the grid is not unified

`PV2_0125` measured the ceiling per dataset and grid. Unifying on 64 is not available:

| dataset | source | grid 64 | consequence |
| --- | ---: | --- | --- |
| Multi-dSprites | 64×64 | the cell **is** the pixel | ceiling exactly 1.0000; required, since grid 32 caps at 0.5619 |
| CLEVR | 240×320 | 3.75 × 5.00 px/cell | possible, data built, but no grid-64 model has ever been trained |
| Tetrominoes | 35×35 | **impossible** | 35 admits no 64 cells; `cell_index` raises. Needs a 128 upscale, which would score against an upscaled truth and so could not sit beside published 35×35 numbers |

Tetrominoes also has little to gain: its grid-32 ceiling is 0.9409 and the model holds
98.63% of it, so a finer grid could add at most 0.059 of ceiling.

## Result 1 — CLEVR, 3 seeds, 1000 test images, pixel protocol

| | FG-ARI | fg-IoU | obj-IoU |
| --- | ---: | ---: | ---: |
| ceiling, grid 32, test 1000 | 0.8406 | 0.8319 | 0.7651 |
| seed 0 (`GC_w0`) | 0.7252 | 0.7315 | 0.6176 |
| seed 1 (`RSC_freeze`) | 0.6351 | 0.6653 | 0.5557 |
| seed 2 (`FRZ_s2`) | 0.6402 | 0.6799 | 0.5632 |
| **mean** | **0.6668** | 0.6922 | 0.5788 |
| std | 0.0506 | 0.0348 | 0.0338 |
| fraction of ceiling | **79.33%** | | |

Patch protocol on the same run, for the appendix: 0.7861 / 0.6920 / 0.6967, mean **0.7249**
± 0.0530, with `pred_fg` 0.1497 / 0.1291 / 0.1328.

The three seeds share seed 0's graph (`PV2_0044`'s freeze), so they are **not three
independent runs** — the same structure the published grid-16 line has, disclosed the same
way.

### This supersedes `BENCHMARK_PLAN.md`'s 0.7142, downward

That figure was **100 validation images with the seed count never stated**. The 3-seed test
figure is **0.6668**, lower by **0.0474**. The fraction of ceiling falls with it, 85.48% →
**79.33%**.

### The comparison claim has to be weakened

| model | published CLEVR pixel ARI-FG | our margin |
| --- | ---: | ---: |
| Rotating Features | 0.65 | **+0.0168 — inside one std (0.0506)** |
| SynCx | 0.59 | +0.0768 |
| CtCAE | 0.54 | +0.1268 |
| CAE++ | 0.27 | +0.3968 |
| CAE | 0.04 | +0.6268 |

`BENCHMARK_PLAN.md` wrote "our 0.7142 sits at the top" of this family. At three seeds on
test, **we are not distinguishable from Rotating Features.** Everything from SynCx down is
cleared with margin.

### A consequence for `ROUTE_MAP_0.9_v4.md`

Its arithmetic used 85.48%. At the corrected **79.33%**, even a lossless grid yields
**0.7933**, and 0.90 would need a ceiling of 1.1345. Route C closes harder, not softer.

## Result 2 — one Tetrominoes seed in three died in training

Grid 32, 300 validation images, seeds trained independently (no freeze, no `--init-from`).

| seed | patch FG-ARI | pixel FG-ARI | pred_fg | spike rate | groups/img | PLV | `spike_plv_collapse` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.9362 | **0.9280** | 0.3004 | — | — | — | 1.6200 |
| 1 | −0.0000 | **0.0000** | 0.0032 | 0.0154 | 0.79 | 0.5657 | **4.9046** |
| 2 | 0.9164 | 0.9126 | 0.2723 | 0.2197 | 8.49 | 0.5355 | 1.8240 |

**Seed 1 is a training failure, not an unlucky seed.** Its spike path is dead: 0.79 groups
per image against seed 2's 8.49, 0.32% of the grid called foreground, and a collapse term
three times the other two.

### There is no 3-seed Tetrominoes mean, and two tempting numbers are both false

| | value | why it must not be reported as the result |
| --- | ---: | --- |
| mean of all three | 0.6135 ± 0.5314 | describes training instability, not segmentation quality |
| mean of the two that trained | 0.9203 (spread 0.0154) | is not a 3-seed mean; selecting the arms that worked |

The honest single sentence is: **two of three independently trained seeds reach 0.91–0.93
pixel FG-ARI; the third's spike path died during training.**

### My own misreading, recorded

I first dismissed `spike_plv_collapse` 4.90 as "not dead, since the dead path reads
9.210340 = log(10⁴)". 9.21 is the saturated value; 4.90 was already fatal. The warning was
there before the evaluation and I passed over it.

## Running, and what it will decide

| job | GPU | script | purpose |
| --- | --- | --- | --- |
| Multi-dSprites grid 64, replicated checkpoint | 2 | `md64c.sh` | whether the 0.2080 at grid 32 was the grid (ceiling 0.5619) or the model |
| Tetrominoes seeds 1–2 with the graph frozen from seed 0 | 3 | `tetfrz.sh` | a valid 3-seed row, at the cost of the seeds not being independent |

`tetfrz.sh` keeps the three independent checkpoints on disk. The record will report the
frozen mean **and** the independent-seed failure rate side by side; the freeze would
otherwise hide a real property of the architecture.

The grid-64 evaluation is the third attempt. Both earlier failures were mine, not the
model's, and `BENCHMARK_EVALUATION.md` §7 names them: a hardcoded batch of 25, then a
training job I placed on the same GPU after calling 12.7 GB safe, whose kernel-level kill
left no traceback while the script's `grep ... || exit 1` swallowed what there was.

## Limitations

Tetrominoes and Multi-dSprites are on 300 **validation** images, not a test split. CLEVR's
three seeds share one graph. The readout thresholds and `--plv-target-density 0.867` were
fixed on CLEVR and carried unchanged, so nothing was tuned on the benchmark datasets, which
is disclosed rather than claimed as a virtue. Published numbers are each paper's own test
split and image count, not ours.

## Next

Read the two running jobs. Then the table: three rows, each with its grid and that grid's
ceiling beside the score, the patch protocol in an appendix, and the Tetrominoes seed
failure reported rather than averaged away.
