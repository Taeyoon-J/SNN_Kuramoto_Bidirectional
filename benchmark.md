# benchmark.md — how to benchmark SW0240, and why these datasets

Written for whoever runs the benchmark on either branch. It answers three things before
any command: **what the model's I/O forces on us**, **whether you need new data or can
take ours**, and **why these datasets and not others**.

Every number quoted below was measured on `patch_v2` and is cited to the test that
produced it, so you can disagree with a specific measurement rather than with the plan.

---

## 1. What the model forces

`predict.py` is strict and says so in the error it raises:

```
input must be nonempty uint8 RGB [B,128,128,3]; images are not resized automatically
```

So, fixed by the package and not negotiable without changing it:

| | |
| --- | --- |
| input | `uint8` `[B, 128, 128, 3]` exactly — no crop, no resize, it raises |
| output | `int64` `[B, 64, 64]`, label 0 is background, positives are object IDs |
| rollout | 1024 steps, first 512 discarded |
| grid | 64 × 64 = 4096 patches, top-k 512 |
| determinism | evaluation image `1320+i` reproduces with inference seed `193000+i` |

**Two consequences that drive everything else.** Every dataset has to be brought to
128×128 on the way in, and every ground truth has to be brought to a 64×64 label grid on
the way out. Those two conversions are where a benchmark gets quietly unfair, so section 3
fixes them per dataset and section 4 says how to check them.

---

## 2. Do you need new data? No — take ours.

Everything below is already built, verified and sitting on the shared server at
`/work/USERS/tkim1/mod/`. Re-deriving it is not just wasted time: the mask axis order
differs by dataset and reading it wrong produces targets that look fine and are noise
(section 4).

| dataset | source | size | objects | what exists |
| --- | --- | --- | --- | --- |
| **Tetrominoes** | `multi_object_datasets`, `tetrominoes_train.tfrecords` | 35×35 | 3 | 8000 PNGs, patch targets at grid 16 and 32, γ, encoder, manifest |
| **Multi-dSprites** | `multi_dsprites_colored_on_colored.tfrecords` | 64×64 | ≤4 | same |
| **CLEVR (ours)** | `clevr_with_masks` | 240×320 | ≤10 | contract v1 (grid 16), v2 (grid 32), v3 (grid 64) |

Per dataset there is:

```
/work/USERS/tkim1/mod/
  images_<ds>/                     8000 PNGs, "<ds>_%06d.png", row order
  images_<ds>.pt                   the same as a uint8 tensor [8000, H, W, 3]
  pixel_labels_<ds>.pt             pixel instance labels [8000, H, W]
  targets_<ds>_grid{16,32}.pt      patch labels, purity, ties, names
  gamma_<ds>_grid{16,32}.pt        [8000, 8, grid^2], mean 0, std 1
  input_encoder_<ds>_k8.pt         224 params, unsupervised, TRAIN split only
  split_manifest_<ds>.json         ranges AND splits (filename lists)
```

**Split is the same shape as CLEVR's**, so nothing about the protocol changes between
datasets: train `[0, 6000)`, validation `[6000, 7000)`, test `[7000, 8000)`.

**The encoders are per dataset and unsupervised.** 8 kernels of 3×3, 224 parameters,
trained by reconstruction on the TRAIN split only — no mask is read. That follows the
precedent `PV2_0086` set for MOVi-A rather than borrowing CLEVR's encoder, because
`PV2_0085` measured what borrowing costs: its control reached 0.3042 against 0.7250
when γ was raw RGB patch means instead of a trained encoder.

Reconstruction loss, TRAIN split: Tetrominoes 0.016649 → **0.000571**, Multi-dSprites
0.043628 → **0.002563**.

**What is not built yet**, and what you would add if you want it: CLEVRTex with its CAMO
and OOD generalisation sets. That is the texture-rich benchmark where slot models
degrade, and it is the strongest place for a binding argument — but it is also a separate
download and conversion.

---

## 3. Why these datasets

### 3a. Because this model's output resolution matches or beats their input resolution

The model emits a 64×64 label grid. Against each dataset's native resolution:

| dataset | native | one output cell covers | effect |
| --- | --- | --- | --- |
| **Tetrominoes** | 35×35 | **0.55 × 0.55 source px** | the grid is *finer* than the image; quantisation is zero |
| **Multi-dSprites** | 64×64 | **1.00 × 1.00 source px** | one cell per source pixel; quantisation is zero |
| CLEVR | 240×320 | 3.75 × 5.00 source px | real quantisation |

This matters because of a measurement on `patch_v2` (`PV2_0119`, `BENCHMARK_PLAN.md`):
quantise the **pixel** ground truth to a patch grid and expand it straight back, and a
*perfect* patch segmenter — zero model error — scores only

| CLEVR, pixel protocol | perfect patch model |
| --- | ---: |
| grid 16 | **0.6973** FG-ARI |
| grid 32 | **0.8355** FG-ARI |

**That ceiling is CLEVR's large images, not the architecture.** On Tetrominoes and
Multi-dSprites the ceiling is effectively 1.0, so a pixel-protocol number from them is a
measurement of the model rather than of the grid. That is the whole reason to run these
two before arguing about CLEVR.

Measured patch purity backs it: grid 32 purity is **0.9941** on Tetrominoes and **0.9878**
on Multi-dSprites against **0.9170** on CLEVR.

### 3b. Because the comparison family publishes on them

The models this architecture actually competes with are the synchrony-binding line, not
the slot line:

| | Tetrominoes | dSprites | CLEVR |
| --- | ---: | ---: | ---: |
| CAE | 0.00 | 0.05 | 0.04 |
| CAE++ | 0.78 | 0.51 | 0.27 |
| CtCAE | 0.84 | 0.56 | 0.54 |
| Rotating Features | 0.42 | 0.84 | 0.65 |
| SynCx | **0.89** | 0.82 | 0.59 |

(pixel ARI-FG, from SynCx, arXiv 2405.17283. Tetrominoes 35×35, dSprites 64×64, CLEVR
filtered 96×96.)

AKOrN (ICLR 2025) is the closest prior work — Kuramoto oscillators for object binding —
and reports **FG-ARI 88.5 ± 0.9, MBO 59.7 ± 0.9** on CLEVRTex at 128×128.

**Do not mix these with slot-family numbers.** Slot Attention reports ARI-FG 99.5 /
91.3 / 98.8 on Tetrominoes / Multi-dSprites / CLEVR**6**, but CLEVR6 caps at six objects
and uses decoder reconstruction. A table with both lines in one column is wrong.

### 3c. Because the metric our result is strong in is the one they report

**MBO (mean best overlap)** is, up to the matching rule, `patch_matched_object_iou`. The
recent papers report it precisely because ARI-FG is insensitive to over-prediction —
`PV2_0090` found Slot Attention calling **85% of patches foreground** against an 11.8%
target and still scoring 0.8102 FG-ARI, while its IoUs sat at 0.1151 / 0.1025.

`PV2_0119` measured our own readout doing the same trade: spectral *k* raises FG-ARI to a
peak of **0.7862 at k=20**, and at that peak predicted foreground is **0.8391** and
foreground IoU has collapsed from 0.7158 to 0.1516. **Report `pred_fg` on every row** or
the table is misleading in our favour.

---

## 4. The two conversions, and how to check them

### Images in: 35×35 or 64×64 → 128×128

Both are integer upscales — Tetrominoes ×3.657 (not integer, so use nearest to avoid
inventing edges), Multi-dSprites exactly ×2. **Use nearest-neighbour, not bilinear.**
Bilinear invents intermediate colours at object boundaries, which is exactly the signal
the γ encoder keys on: `PV2_0115` measured that a patch's γ deviation from its 5×5
neighbourhood separates true foreground at **AUC 0.947**, and smoothing the boundary
degrades it.

### Ground truth out: pixel labels → 64×64

Use the same modal vote the contract uses (`make_targets_grid.py` / `build_mod_targets.py`),
on the **pixel** labels, at grid 64. For Tetrominoes and Multi-dSprites this is lossless
or near it by section 3a.

### Two checks that are not optional

**Mask axis order differs by dataset.** Measured, by how often neighbouring pixels share
a label — a real segmentation is well above 0.9:

| dataset | EHW | HWE | correct |
| --- | ---: | ---: | --- |
| multi_dsprites | 0.0349 | **0.9764** | **HWE** |
| tetrominoes | **0.9384** | 0.8490 | **EHW** |
| CLEVR | **0.9926** | 0.9796 | **EHW** |

Reading a channels-last mask as channels-first leaves a period-E stripe in the labels and
**raises no error**. The first Multi-dSprites build here had patch purity 0.2813 against
0.2 for a random 5-way assignment and looked like a normal file. `build_mod_targets.py`
now refuses to save below a neighbour-coherence of 0.85.

**Run the preflight.** `collaborative_test/preflight.py` compares a dataset's targets and
manifest against the CLEVR reference in the order the evaluator touches the fields, and
exits non-zero. It exists because four consecutive evaluation failures here were all one
question asked four times — missing `contract_version`, `names`, `patch_size`, and the
manifest's `splits`.

The `splits` check is the one that matters beyond convenience: the evaluator compares the
targets' `names` to `manifest["splits"][split]`, which is **how the γ tensor and the
ground truth are verified to describe the same images in the same order**.

```bash
python collaborative_test/preflight.py --dataset tetrominoes --grid 32   # 8/8 expected
```

---

## 5. Protocol

**Main table: pixel protocol.** Every published number in this field is per pixel. A
patch-protocol table can cite nothing and would need every baseline re-run here. Report
**FG-ARI and MBO together**, with `pred_fg` beside them.

**Appendix: the patch contract**, with the grid ceiling stated — 0.6973 at grid 16 and
0.8355 at grid 32 on CLEVR — so a reader can see what the grid costs separately from what
the model costs.

**Three independently trained seeds**, per metric, with standard deviation. A single
checkpoint evaluated three times is one checkpoint and must be labelled as one. That rule
already stands in `baselines/slot_attention.md` and both branches apply it.

**Re-run at least one baseline yourself per dataset** and check it reproduces the
published value. If our Slot Attention does not reproduce published Slot Attention, the
harness is wrong — which is exactly what `PV2_0090` caught, and it cost a claimed win.

### Two columns the table needs or the numbers mislead

Checked in AKOrN's `eval_obj.py` rather than inferred: its masks come from agglomerative
average-linkage clustering of the **oscillator states**, and the **cluster count is given
per dataset** (`n_clusters = 11` for CLEVR, `4` for Tetrominoes, `7` for dSprites/COCO).

| | cluster count | read from |
| --- | --- | --- |
| Slot Attention, SLATE, DINOSAUR | fixed per dataset | slot masks |
| CAE, CtCAE, Rotating Features, SynCx | fixed per dataset | complex phase |
| AKOrN | fixed per dataset (11 / 4 / 7) | oscillator states |
| **SW0240 / our components readout** | **none given — falls out of connectivity** | **spikes** |

Both columns favour us and both have to be stated, not implied.

### What must be disclosed about SW0240

**It is mask-supervised.** Its own README says so: *"240번은 mask-supervised입니다."* Its
three-seed means are **0.889187 / 0.754775 / 0.686438**, and the endpoint is
`SW0249_matched35000_terminal` — 70,000 images of training followed by 35,000 × 10 in the
final four-model comparison.

A table that compares a mask-supervised model to unsupervised baselines without saying so
is not a fair table. Either mark the row, or run the unsupervised line beside it.

---

## 6. Running it

### Get the weights

The checkpoint is **not in git** — `checkpoints/manifest.json` records
`sw240_seed0.pt`, 201,791,722 bytes, sha256 `ba79c452…fced06`, model sha256
`1e2def29…5b03a4`. Export once from the research server:

```bash
python tools/export_from_research.py --legacy-root /path/to/patch_v2_sw \
  --output checkpoints --seed 0        # --seed 1 / 2 for the others
```

### Predict

```bash
python predict.py --checkpoint checkpoints/sw240_seed0.pt \
  --input images_128.npy --output outputs/masks.npz --device cuda:0
```

`images_128.npy` is `uint8 [B,128,128,3]`, built by nearest-neighbour upscaling from
`/work/USERS/tkim1/mod/images_<ds>.pt` in row order, so row *i* is
`<ds>_%06d.png` and lines up with `split_manifest_<ds>.json`.

### Score

`snn_kuramoto_bidirectional/evaluation.py` in this package carries
`patch_fg_ari`, `patch_foreground_iou`, `patch_matched_object_iou` and
`evaluate_patch_masks` — the same functions both branches score with, so the comparison
does not depend on whose harness ran.

For the pixel protocol, expand the 64×64 prediction to the source resolution with the
contract's own cell edges and score against the **pixel** labels. On Tetrominoes and
Multi-dSprites that expansion is lossless or near it (section 3a); on CLEVR it is not, and
the ceiling above is what it costs.

---

## 7. Open questions, stated rather than hidden

1. **Nobody has run SW0240 on anything but CLEVR.** Whether its spike path survives a new
   dataset is unknown. `PV2_0094` measured that an absolute membrane threshold does not
   transfer across oscillator counts; nothing says it transfers across datasets. Our own
   Tier 1 run carried `--membrane-threshold-mode population` for that reason and the path
   held, at FG-ARI 0.9445 on Tetrominoes — **patch contract, one seed, 100 images, and
   predicted foreground 0.2984 against a true 0.2363**, so read it as "it runs", not as a
   score.
2. **The 128×128 input is a constraint, not a choice.** Upscaling a 35×35 image to 128×128
   gives the model more pixels than the data has information, and whether that helps or
   hurts is untested.
3. **CLEVRTex is where the argument is strongest and it is not prepared.** AKOrN's 88.5
   is there.
