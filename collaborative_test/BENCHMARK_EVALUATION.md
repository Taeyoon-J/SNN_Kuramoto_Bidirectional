# How we evaluate on the benchmark datasets

The procedure, the two protocols, and the numbers it has produced. Companion to
`evaluation_contract.md`, which fixes the metric definitions, and to `benchmark.md` on
`patch_final`, which tells the peer which data to use and why.

Every number here was measured; where a number is older or weaker, the row says so.

---

## 1. Two protocols, and which one a table may use

| | patch protocol | pixel protocol |
| --- | --- | --- |
| scored over | patch cells | image pixels |
| defined by | `evaluation_contract.md` | the published literature |
| our use | internal comparisons, every `PV2_` record | **the paper table** |
| conversion | ground truth → patch grid by modal vote | prediction → pixels by the contract's own cell edges |

**A paper table has to be pixel-protocol.** Every model this architecture competes with —
CAE, CAE++, CtCAE, Rotating Features, SynCx — publishes per pixel, so a patch-protocol
table can cite nothing and has to re-run all of them. That was decided in
`BENCHMARK_PLAN.md` and the patch table goes to the appendix with its ceiling stated.

`evaluation_contract.md` forbids expanding a predicted mask to pixels to score it, and that
still holds **for contract scores**. The pixel conversion is a separate, declared protocol
for external comparability, not a contract score, and it is labelled as such everywhere.

The three metrics are the same functions in both: `patch_fg_ari`, `patch_foreground_iou`,
`patch_matched_object_iou` from `collaborative_test/evaluation.py`, used unchanged.

---

## 2. Measure the grid's ceiling first. Never infer it.

Quantise the **pixel** ground truth to the patch grid, expand it straight back, score it
against the original. That is what a perfect patch segmenter scores: zero model error, pure
quantisation. `benchmark/ceil.py`, CPU only, no model.

```
python benchmark/ceil.py          # prints every dataset x grid
```

| dataset | source | grid | source px/cell | ceiling FG-ARI | fg-IoU | obj-IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tetrominoes | 35×35 | 16 | 2.19 | 0.6881 | 0.7705 | 0.7667 |
| Tetrominoes | 35×35 | **32** | 1.09 | **0.9409** | 0.9602 | 0.9587 |
| Multi-dSprites | 64×64 | 16 | 4.00 | 0.4740 | 0.7520 | 0.6980 |
| Multi-dSprites | 64×64 | 32 | 2.00 | 0.5619 | 0.8699 | 0.8368 |
| Multi-dSprites | 64×64 | **64** | 1.00 | **1.0000** | 1.0000 | 1.0000 |
| CLEVR | 240×320 | 16 | 15.0 × 20.0 | 0.6973 | 0.6965 | 0.5814 |
| CLEVR | 240×320 | **32** | 7.5 × 10.0 | **0.8406** | 0.8319 | 0.7651 |

CLEVR grid 32 is on the 1000 test images; its earlier 0.8355 was 100 validation images.
CLEVR grid 64 is **unmeasured** — see §7.

### Which grid each dataset must use, and why

- **Multi-dSprites: grid 64.** The cell is the pixel, the conversion is the identity, the
  ceiling is exactly 1.0000. At grid 32 the ceiling is **0.5619**, below CtCAE's published
  0.56 and far below SynCx's 0.82 — the protocol decides the comparison before the model
  runs, and a loss there means nothing.
- **Tetrominoes: grid 32, and no finer.** 35 does not admit 64 cells; some cell would be
  empty and `make_targets_grid.py` raises. Every Tetrominoes number reads against 0.9409.
- **CLEVR: grid 32 or finer.** Grid 16's 0.6973 ceiling is below what the patch score
  already reports, which is why the patch and pixel protocols rank the two grids oppositely.

### Do not estimate a ceiling from patch purity

This cost a run and a wrong claim in two documents. Grid-32 purity is 0.9941 on Tetrominoes
and 0.9878 on Multi-dSprites — within 0.006 of each other, reading as "both nearly
lossless". Their measured ceilings are **0.9409 and 0.5619**, 0.379 apart.

Purity is a per-cell label-agreement rate. FG-ARI counts **pairs**, so a one-cell boundary
error on a small object removes far more pair agreements than its cell count suggests, and
two objects separated by one cell merge into a single cluster under the quantisation no
matter what the model does. Multi-dSprites has up to five overlapping sprites; Tetrominoes'
three are separated. Measure the ceiling.

---

## 3. The pipeline, in order

Scripts live in `benchmark/` on `patch_final` (copies of `collaborative_test/`).

| # | step | script | per |
| --- | --- | --- | --- |
| 1 | input encoder, 224 params, reconstruction on TRAIN only | `bench_encoder.py` | dataset |
| 2 | pixel masks → patch targets, with the measured mask axis order | `build_mod_targets.py` | dataset × grid |
| 3 | features → γ at unit variance | `build_gamma_grid.py` | dataset × grid |
| 4 | **8 checks in the order the evaluator hits the fields** | `preflight.py` | **every evaluation** |
| 5 | the model, patch predictions saved | `evaluate_model.py --save-predictions` | checkpoint |
| 6 | prediction → pixels, scored beside the ceiling | `pixbench.py` | checkpoint |

### Step 4 is not optional

`preflight.py` exists because **four consecutive evaluations died** on a missing
`--expect-contract`, then a missing `names`, then `patch_size`, then the manifest's
`splits` — each one a ten-hour training run reaching the evaluator and failing on a field.
A fifth instance followed: both arms of `PV2_0121` had trained, 608 and 737 minutes, and
neither had ever been evaluated because `tier1.sh` predates the check. It takes seconds and
exits non-zero.

Run it **unpiped**. `preflight.py ... | tail -12 || exit 1` takes the pipeline's status from
`tail`, which is always 0, so a run continues past every FAIL. That happened here.

### The real commands

Evaluation, grid 32:

```bash
python evaluate_model.py --checkpoint $R/<name>/core.pt \
  --gamma-seq-path $M/gamma_<ds>_grid32.pt --targets $M/targets_<ds>_grid32.pt \
  --manifest $M/split_manifest_<ds>.json --expect-contract v2-pixel \
  --grid 32 --num-regions 1024 --split validation --limit 300 --eval-batch 25 \
  --spike-per-component --geodesic-steps 3 --geodesic-contrast 2.0 \
  --graph-spatial-decay 0.35 --geodesic-chunk 4 \
  --membrane-threshold-mode population --membrane-threshold-k 0.5 \
  --num-time-steps 1024 --settle 512 --synchrony-threshold 0.05 \
  --spatial-sigma 4.0 --min-group-size 2 \
  --save-predictions $M/pred_<ds>_grid32.npz
```

Then the pixel conversion:

```bash
python pixbench.py --dataset <ds> --predictions $M/pred_<ds>_grid32.npz --grid 32
```

At grid 64, change `--grid 64 --num-regions 4096 --geodesic-chunk 2 --eval-batch 4`.

**`--eval-batch` is not free — see §5.** CLEVR uses `--expect-contract 2` and its own
manifest at `collaborative_test/data/split_manifest.json`.

---

## 4. What is read, and when

- **Thresholds and hyperparameters are selected on validation.** The readout settings
  (`--synchrony-threshold 0.05 --spatial-sigma 4.0 --min-group-size 2`) and
  `--plv-target-density 0.867` were fixed on CLEVR and carried to the benchmark datasets
  **unchanged**, so nothing was tuned on Tetrominoes or Multi-dSprites. Disclosed, not
  claimed as a virtue.
- **Test is read once per settled setting.** CLEVR's benchmark row uses the 1000 test
  images; Tetrominoes and Multi-dSprites are still on 300 validation images and are labelled
  as such.
- **Three seeds decide anything.** One seed is exploration and is reported as one seed.
- **Oracle results are separate** and never compared to another model.

---

## 5. The evaluation batch size changes the score

`membrane_layer._threshold`, in the `population` mode every benchmark number uses:

```python
ref = mem.detach()
return ref.mean() + self.threshold_k * ref.std().clamp_min(1e-6)
```

`mean()` and `std()` reduce over the **whole tensor, batch dimension included**, so the
firing threshold is one scalar per forward pass — and a forward pass is a batch. Two runs of
the same checkpoint on the same images with different `--eval-batch` can score differently,
and image *i*'s mask depends on which images shared its pass.

**Therefore: a score is only reproducible together with its batch size.** Every recorded
number used 25, which is the default. Grid 64 cannot use 25 — it OOMs, because the per-node
tensors grow 4× with the node count while the coupling matrix grows 16× — so a grid-64
number is **not strictly comparable** to a grid-32 one and that has to be said.

The size of the effect is **unmeasured**. `PV2_0126` records the repair (one threshold per
image) as a core change needing its own ablation, not a quiet edit.

---

## 6. Results so far

### CLEVR, 3 seeds, 1000 test images

| protocol | grid | FG-ARI | fg-IoU | obj-IoU | ceiling FG-ARI | fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| patch | 16 (`PV2_0076`) | **0.7603** ± 0.0003 | 0.3373 | 0.6598 | — | — |
| patch | 32 (`PV2_0104`) | 0.7249 ± 0.0530 | 0.7133 | 0.6580 | — | — |
| **pixel** | **32** | **0.6668** ± 0.0506 | 0.6922 ± 0.0348 | 0.5788 ± 0.0338 | 0.8406 | **79.33%** |

Per seed, pixel: 0.7252 / 0.6351 / 0.6402. The three share seed 0's graph (`PV2_0044`'s
freeze), so they are **not three independent runs** — the same structure the published
grid-16 line has.

The patch and pixel protocols rank the grids **oppositely**: grid 32 costs −0.0354 of patch
score but gains pixel score, because its ceiling rises further than its score falls.

An earlier pixel number, **0.7142**, appears in `BENCHMARK_PLAN.md`. It was 100 validation
images with the seed count never stated. The 3-seed test figure is **0.6668**, −0.0474 lower.

### Tetrominoes, grid 32, 300 validation images

| seed | patch FG-ARI | pixel FG-ARI | pred_fg | spike rate | groups/img |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.9362 | **0.9280** | 0.3004 | — | — |
| 1 | −0.0000 | 0.0000 | 0.0032 | 0.0154 | 0.79 |
| 2 | 0.9164 | 0.9126 | 0.2723 | 0.2197 | 8.49 |

**Seed 1 is a training failure, not a bad seed.** Its spike path is dead:
`spike_plv_collapse` 4.90 against 1.62 and 1.82 for the others. There is **no 3-seed
Tetrominoes mean**: two seeds reach 0.91–0.93 and the third died. The arithmetic mean
including it is 0.6135 ± 0.5314, which describes training instability, not model quality,
and reporting either that or the two-seed 0.9203 as a 3-seed result would be false.

`PV2_0044`'s graph freeze is the in-project fix, already validated on CLEVR; using it means
disclosing that the Tetrominoes seeds share a graph too.

### Multi-dSprites, grid 32, 300 validation images, seed 0

| protocol | FG-ARI | fg-IoU | obj-IoU | ceiling | fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| patch | 0.2363 | 0.5208 | 0.4584 | — | — |
| pixel | 0.2080 | 0.5334 | 0.4180 | 0.5619 | 37.02% |

`pred_fg` 0.0927, 18.57 groups per image. **Two separate losses that must not be merged:**
the grid caps this at 0.5619, so no win was available at grid 32; and the model is a further
0.3539 under even that cap, *under*-painting foreground where Tetrominoes over-paints.

### Against the published synchrony family, pixel ARI-FG

| model | Tetrominoes | Multi-dSprites | CLEVR |
| --- | ---: | ---: | ---: |
| CAE | 0.00 | 0.05 | 0.04 |
| CAE++ | 0.78 | 0.51 | 0.27 |
| CtCAE | 0.84 | 0.56 | 0.54 |
| Rotating Features | 0.42 | 0.84 | 0.65 |
| SynCx | 0.89 | 0.82 | 0.59 |
| **ours** | **0.9280** (1 seed) | 0.2080 (1 seed, wrong grid) | **0.6668** ± 0.0506 (3 seeds) |

Read honestly:

- **Tetrominoes 0.9280 is above every published number**, and sits 0.0129 from its grid's
  ceiling. One seed, validation, and the same run's fg-IoU is **0.2561 under** the ceiling
  where its FG-ARI is 0.0129 under it — the model over-paints and FG-ARI, scored over
  true-foreground cells only, cannot see it. **FG-ARI alone would be a selected number.**
- **CLEVR 0.6668 beats SynCx and below, but is not distinguishable from Rotating
  Features' 0.65**: the margin is +0.0168 against a std of 0.0506.
- **Multi-dSprites loses**, at a grid where winning was impossible.

---

## 7. Not done

| | state |
| --- | --- |
| CLEVR pixel ceiling at grid 64 | **unmeasured**; `PV2_0120`'s 0.9890 is purity, which §2 shows does not bound it. CPU-only, and the cheapest decisive number left |
| Multi-dSprites at grid 64 | not measured yet, twice for my reasons: the first run OOMed on the hardcoded batch 25, and the second was progressing at 5.5 GB until I launched a training job onto the same GPU and called 12.7 GB safe -- one of the two was killed by the kernel, which leaves no traceback, and the script piped its output through `grep ... \|\| exit 1` so the empty match exited silently. Re-run alone, unfiltered |
| Tetrominoes 3 seeds | 2 of 3; seed 1 needs retraining |
| CLEVR pixel at grid 16, 3 seeds on test | only the 100-image figure exists |
| Slot Attention through this pipeline on these datasets | not run; the table currently cites published numbers instead |
| CLEVRTex, CAMO/OOD | not downloaded |

---

## 8. Data and access

Full inventory in `benchmark.md` on `patch_final`. Everything is under
`/work/USERS/tkim1/mod/` (`drwxr-xr-x`, files `-rw-r--r--`), reachable through
`/work/USERS/tkim1` (`drwxr-x--x tkim1:acm`) — the peer is in `acm`, so read access is
confirmed. CLEVR's manifest is the exception, at
`/export_home/tkim1/collaborative_test/data/split_manifest.json`, on the mount that has
faulted before; copy it rather than depend on it.

**`mod/` contains two Multi-dSprites families.** `mdsprites` (short name) is the early build
whose mask axis order was wrong — `[H,W,E]` read as `[E,H,W]`, giving a period-5 stripe and
target purity 0.2813 against a chance level of 0.2, with **no error raised**. Use
`multi_dsprites` only. `build_mod_targets.py` now asserts spatial coherence so it cannot
recur.
