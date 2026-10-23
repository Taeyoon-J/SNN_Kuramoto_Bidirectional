# PV2_0125 — the pixel protocol, and the grid ceiling measured instead of assumed

- **status** completed (grid-32 arms); grid-64 Multi-dSprites arm running
- **parent** `PV2_0121` (Tier 1 training), `BENCHMARK_PLAN.md`
- **branch** `patch_v2`; `benchmark.md` and the script bundle shipped on `patch_final` (`f9301ed`)
- **data** `tetrominoes`, `multi_dsprites`, 300 validation images each (rows 6000–6299)
- **contract** `v2-pixel`; metrics from `collaborative_test/evaluation.py`, unchanged
- **seeds** 1 (seed 0). **Not a 3-seed result and not a test-split result.**

## Hypothesis

`PV2_0121` produced patch-protocol numbers. Every number this architecture competes with —
CAE, CAE++, CtCAE, Rotating Features, SynCx — is published per **pixel**, so a patch number
cannot be put in the same table. Converting ours to pixels should make the comparison legal,
and the grid's own cost should be small on these two datasets because their images are small.

**The second half of that hypothesis was wrong, and that is the finding.**

## What happened first: `PV2_0121`'s evaluations had never run

Both arms of `tier1.sh` died at the evaluator:

```
ValueError: targets declare contract version 'v2-pixel' but --expect-contract is '1'.
```

The training had completed — Tetrominoes 608 min, Multi-dSprites 737 min — and both
checkpoints were on disk. Only the evaluation was missing. This is the fifth instance of
the failure class `preflight.py` was written for, and `tier1.sh` predates it.

## Result 1 — the grid ceiling, measured

`benchmark/ceil.py`: quantise the pixel ground truth to a grid, expand it straight back,
score it. No model. This is what a **perfect** patch segmenter scores.

| dataset | source | grid | source px / cell | FG-ARI | fg-IoU | obj-IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tetrominoes | 35×35 | 16 | 2.19 | 0.6881 | 0.7705 | 0.7667 |
| tetrominoes | 35×35 | 32 | 1.09 | **0.9409** | 0.9602 | 0.9587 |
| multi_dsprites | 64×64 | 16 | 4.00 | 0.4740 | 0.7520 | 0.6980 |
| multi_dsprites | 64×64 | 32 | 2.00 | **0.5619** | 0.8699 | 0.8368 |
| multi_dsprites | 64×64 | 64 | 1.00 | **1.0000** | 1.0000 | 1.0000 |

`BENCHMARK_PLAN.md` and the first `benchmark.md` both asserted the ceiling on these two
datasets was "effectively 1.0". It is 0.9409 and 0.5619. **The assertion was inferred from
patch purity** — 0.9941 and 0.9878 at grid 32 — and purity does not predict FG-ARI. Purity
is a per-cell label-agreement rate; ARI counts *pairs*, so a one-cell boundary error on a
small object removes many more pair agreements than its cell count suggests, and two
sprites separated by one pixel merge into a single cluster under the quantisation no matter
what the model does. Multi-dSprites has up to five sprites that overlap; Tetrominoes'
objects are separated. That is the whole difference between 0.5619 and 0.9409 at the same
~1–2 px per cell.

**Consequence, binding on the protocol:** at grid 32 a *perfect* model scores 0.5619 on
Multi-dSprites, below CtCAE's published 0.56 and far below SynCx's 0.82. The protocol
decided that comparison before the model ran.

## Result 2 — the two Tier 1 models under the pixel protocol

Grid 32, seed 0, 300 validation images, via `benchmark/pixbench.py`.

| | Tetrominoes | Multi-dSprites |
| --- | ---: | ---: |
| ceiling at grid 32 | 0.9409 | 0.5619 |
| **pixel FG-ARI** | **0.9280** | **0.2080** |
| pixel fg-IoU | 0.7041 | 0.5334 |
| pixel obj-IoU | 0.7889 | 0.4180 |
| gap to ceiling, FG-ARI | 0.0129 | 0.3539 |
| predicted foreground fraction | 0.3004 | 0.0927 |
| groups per image | — | 18.57 |
| patch FG-ARI (for reference) | 0.9362 | 0.2363 |

Published pixel ARI-FG, same datasets: CAE .00/.05, CAE++ .78/.51, CtCAE .84/.56,
Rotating Features .42/.84, SynCx .89/.82.

### Tetrominoes 0.9280 is above every published number in the family

And it sits **0.0129** from the grid's own ceiling, so the model is very nearly saturating
what grid 32 can express on this dataset. One seed, validation split, 300 images.

### The same run's other two columns are not close to that ceiling

fg-IoU is **0.2561** under the ceiling where FG-ARI is 0.0129 under it, with predicted
foreground at 0.3004 against a true foreground of 0.2363. The model over-paints, and FG-ARI
cannot see it because FG-ARI is scored over **true-foreground** cells only. A Tetrominoes
table reporting FG-ARI alone would be a selected number. This is the same asymmetry
`PV2_0114` measured on CLEVR — inventing foreground costs 4.7× what missing it does — and
it is the reason the contract requires all three columns together.

### Multi-dSprites loses, for two reasons that must not be merged

The grid caps it at 0.5619, so no win was available at grid 32; **and** the model is a
further 0.3539 under even that cap. Predicted foreground is 0.0927 against Tetrominoes'
0.3004 — here it *under*-paints, the opposite failure, on a dataset with low-contrast
sprites on varied backgrounds and a 224-parameter reconstruction encoder. Fixing the grid
will not by itself fix the model.

## Is the comparison legal? Yes — no mask supervision anywhere

Checked rather than assumed, by reading `tier1.sh`:

- training receives `--gamma-seq-path` only; `--targets` is passed to `evaluate_model.py`
  and **never** to training
- the loss terms are `--spike-plv-weight 5.0 --plv-collapse-weight 1.0
  --plv-bimodality-weight 6.0 --plv-balance-weight 10.0 --plv-coherence-weight 0.5`, all
  PLV-structure terms on the model's own phases and spikes; `--structural-weight 0`
- the per-dataset input encoders are 224 parameters trained by reconstruction on TRAIN only
- the readout is connected components over spike synchrony, so **the cluster count is not
  supplied**. Several models in the comparison family are given K

The one scalar carried from CLEVR tuning is `--plv-target-density 0.867`, and the
evaluation thresholds (`--synchrony-threshold 0.05 --spatial-sigma 4.0 --min-group-size 2`)
were carried over unchanged rather than tuned on these datasets. Both are disclosed.

## Failures inside this test, and their fixes

1. **`build_mod_targets.py` clobbered the manifest.** Rebuilding at grid 64 rewrote
   `split_manifest_multi_dsprites.json` from `ranges` alone and dropped the `splits` that
   `fixman.py` had added afterwards — breaking grid 32, which was already finished. Fixed
   at the root: `names`, `patch_size` and `splits` are now written **by the builder**, with
   an assert that the named image files exist, so `addnames.py`/`fixman.py` are no longer a
   separate step that a rebuild can undo. Both datasets rebuilt; all five
   dataset×grid configurations now pass preflight 8/8.
2. **My runner swallowed preflight's exit code.** `preflight.py ... | tail -12 || exit 1`
   takes the pipeline's status from `tail`, which is always 0, so the run continued past
   three FAILs and died at `KeyError: 'names'` after replicating the checkpoint. preflight
   worked; the script ignored it. Pipe removed.

## Artifacts

- `benchmark/ceil.py`, `benchmark/pixbench.py` — new, shipped on `patch_final` (`f9301ed`)
- `/work/USERS/tkim1/mod/pred_{tetrominoes,multi_dsprites}_grid32.npz`
- `/work/USERS/tkim1/runs/T1_{tetrominoes,multi_dsprites}_s0/core.pt`
- `/work/USERS/tkim1/{pixbench.out,pixbench2.out,rebuild.out,md64.out}`

## Limitations

One seed. Validation split, not test. 300 images against the published papers' own test
sets. Multi-dSprites' grid-32 number is not a meaningful comparison and is reported only to
separate the grid's cost from the model's.

## Next

Multi-dSprites at grid 64, where the ceiling is 1.0, by replicating the grid-32 checkpoint
with `map_grid.py` at zero gradient steps (`PV2_0122` showed both IoUs survive that). It
isolates how much of the 0.2080 was the grid. Then 3 seeds on Tetrominoes, and the test
split read once.
