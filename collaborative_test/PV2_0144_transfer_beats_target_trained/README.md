# PV2_0144 — the CLEVR-trained core beats cores trained on the target dataset

**Status** completed. No GPU, no retraining — the predictions were already on disk.
**Supersedes** PV2_0140, whose plan could not work (see the note at the end)
**This is the project's first three-seed table under its own patch protocol.**

## How this became possible without a GPU

The six finished per-dataset cores were only ever scored under the **pixel** protocol, which is
not comparable to anything else here. But their per-image predictions were saved as **32×32
label maps**, and the patch contract scores a grid prediction against truth quantised to the
grid. So the same lift Slot Attention was given at grid 16 and 32 — `np.repeat` to 64, exact for
an integer factor — puts them on our protocol, with the same `evaluate_patch_masks`, the same
targets, and the same image ids.

## The table

Patch protocol, grid 64, the **same 320 evaluation images** as every other row in this project.

| dataset | core trained on | seeds | fg_ari | foreground_iou | matched_object_iou |
|---|---|---|---|---|---|
| **Tetrominoes** | **itself** (3,500×10, grid 32→64) | 3 | 0.7730 ± 0.0396 | 0.7510 ± 0.0434 | 0.7659 ± 0.0220 |
| Tetrominoes | **CLEVR** (10,000×10, grid 64) | 1 | **0.8580** | **0.7885** | **0.8173** |
| **Multi-dSprites** | **itself** (3,500×10, grid 32→64) | 3 | 0.2964 ± 0.0826 | 0.2175 ± 0.0637 | 0.1426 ± 0.0559 |
| Multi-dSprites | **CLEVR** (10,000×10, grid 64) | 1 | **0.6974** | **0.8177** | **0.7628** |

Per-seed values — Tetrominoes fg_ari 0.7812 / 0.8079 / 0.7300; Multi-dSprites 0.2854 / 0.3840 /
0.2199.

## What it says

**A core trained on CLEVR and never shown these datasets beats cores trained on the datasets
themselves, on both of them.**

- **Multi-dSprites: 0.6974 against 0.2964 ± 0.0826 — a gap of 0.40, or 4.9 standard
  deviations.** Not close.
- **Tetrominoes: 0.8580 against 0.7730 ± 0.0396 — a gap of 0.085, 2.1 standard deviations.**

This is the strongest form the generalisation claim has taken, and the first time one side of it
carries an error bar.

## What is confounded, stated plainly

The target-trained cores differ from ours in **two** ways besides what they trained on:

| | target-trained | ours |
|---|---|---|
| grid | 32, lifted to 64 | native 64 |
| budget | 3,500 × 10 (2,190 updates) | 10,000 × 10 (6,250 updates) |

So "trained on CLEVR" is not the only candidate explanation; grid and budget are confounded with
it. The gap on Multi-dSprites is 4.9σ, which is hard to attribute entirely to those, but the
honest statement is that this comparison **narrows** rather than settles the question.

**What settles it is already running.** `PV2_0138` trains cores on each dataset at **grid 64,
10,000 × 10** — matching ours on both confounds — and `chain261.sh` will score them on these
same 320 images. About 14 h out. At that point the only difference between the two rows is what
the core trained on.

Our own side also remains **seed 0**. The ± above is on the target-trained side only.

## What it does to our head gains

This std is now measured **under our own protocol on our own 320 images**, so the caveat
attached to PV2_0143's numbers (different grid *and* protocol) is much weaker here — only grid
and budget differ:

| dataset | our reported head gain (fg_ari) | measured seed std | ratio |
|---|---|---|---|
| Tetrominoes | +0.0006 | **0.0396** | **66× smaller** |
| Multi-dSprites | +0.0067 | **0.0826** | **12× smaller** |

**The head gains on these two datasets are not measurable at one seed.** PV2_0141 said this from
epoch and arm wobble; this says it from actual training seeds, on the right protocol. Only
CLEVR's gain (+0.0086 / +0.0285 / +0.0298) is of a size that could survive, and that is still
unproven.

## Why PV2_0140's plan could not have worked

`PV2_0140` proposed getting a three-seed table by caching those cores with `cache_any.py` and
refitting the head. That is impossible: the cores are **grid 32** —
`wrapped.core.kuramoto.omega` is `(1024, 4)` — and `cache_any.py` is `GRID = 64`, which builds a
model expecting `(4096, 4)`. The `state_dict` load would fail outright.

This route avoids the problem entirely by scoring predictions that already exist, at the cost of
not being able to refit a head onto those cores. That is why the table above is **core-only** on
both sides, which is in any case the cleaner comparison: `PV2_0143` and this record both show
the head's contribution on these datasets is inside the noise.

## Reproduce

```bash
python3 score3seed.py --code benchmark_codes --config benchmark_codes/configs/frontier.json \
  --run <run> --limit 320 --out fit3500/patch_3seed_320.json
```

Pure numpy and the benchmark's own evaluation module; no model is loaded.
