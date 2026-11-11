# Results tables

Every number: our evaluation function (`evaluate_patch_masks`), patch contract at grid 64, the
same reserved 320 image ids per dataset, background id 0, predictions never expanded to pixels.
**AG-ARI is `patch_fg_ari`** (foreground ARI on the patch grid). **One core seed throughout**, by
decision — no 3-seed mean exists anywhere in this project.

`-1` means **not measured**, not zero.

Our rows are the **253 core alone, no head**, which is what the baselines are being compared
against. Head rows are in `PV2_0146` / `PV2_0153` / `PV2_0156`.

---

## 1. Interusability among data — every model trained with CLEVR 10,000 × 10

100,032 exposures each (CAE family 100,032 at batch 64; see `PV2_0151` for each model's budget
verdict). Tetrominoes and Multi-dSprites are **zero-shot** for every row.

| | CLEVR AG-ARI | CLEVR FG-IoU | CLEVR Obj-IoU | Tetro AG-ARI | Tetro FG-IoU | Tetro Obj-IoU | mdS AG-ARI | mdS FG-IoU | mdS Obj-IoU |
|---|---|---|---|---|---|---|---|---|---|
| **253** | **0.8421** | **0.7545** | **0.6410** | **0.8580** | **0.7885** | **0.8173** | **0.6974** | **0.8177** | **0.7628** |
| Slot Attention | 0.2872 | 0.2220 | 0.0801 | 0.1739 | 0.2192 | 0.1071 | 0.1930 | 0.0876 | 0.0680 |
| AKOrN (conv) | 0.6046 | 0.4219 | 0.2769 | 0.7202 | 0.7540 | 0.5913 | 0.4055 | 0.6834 | 0.5434 |
| AKOrN (attn) | 0.6027 | 0.3327 | 0.3750 | 0.5676 | 0.7383 | 0.4687 | 0.4343 | 0.6019 | 0.5751 |
| AKOrN attn + up-tiling ×4 | 0.6326 | 0.3468 | 0.3636 | 0.5484 | 0.7230 | 0.4167 | 0.5505 | 0.6539 | 0.6348 |
| ItrSA | 0.5794 | 0.2787 | 0.3853 | 0.6319 | 0.7137 | 0.5687 | 0.4817 | 0.2489 | 0.6093 |
| CAE++ | 0.2838 | 0.2193 | 0.0882 | 0.2715 | 0.2452 | 0.1901 | 0.2648 | 0.1048 | 0.1125 |
| CtCAE | 0.2357 | 0.2148 | 0.0964 | 0.2681 | 0.2420 | 0.2034 | 0.3079 | 0.1158 | 0.1609 |
| CAE | 0.1091 | 0.2008 | 0.0634 | 0.1526 | 0.2305 | 0.1192 | 0.1200 | 0.1038 | 0.0630 |
| **best baseline per cell** | 0.6326 | 0.4219 | 0.3853 | 0.7202 | 0.7540 | 0.5913 | 0.5505 | 0.6834 | 0.6348 |
| **our margin** | +0.2095 | +0.3326 | +0.2557 | +0.1378 | **+0.0345** | +0.2260 | +0.1469 | +0.1343 | +0.1280 |

Nine of nine against a best-per-cell composite of three different models. Narrowest cell:
Tetrominoes Foreground IoU, +0.0345. The CAE family's cluster count was an **oracle** in their own
code (per-image true object count into KMeans); it was removed before these rows were produced —
`PV2_0151`.

---

## 2. Optimal timestep — model 253, trained with CLEVR 10,000 × 10, evaluated with CLEVR

Core alone. `settle = T // 2` at every T (it was hardcoded at 512 of 1024, the second half).

| time step | AG-ARI | Foreground IoU | Object IoU | Δ AG-ARI | s / frame |
|---|---|---|---|---|---|
| **1024** | 0.8421 | 0.7545 | 0.6410 | — | 10.8 |
| **512** | **0.8402** | **0.7594** | 0.6340 | **−0.0019** | 10.0 |
| 256 | 0.8159 | 0.7358 | 0.6084 | −0.0262 | 8.8 |
| 128 | 0.5796 | 0.4988 | 0.3985 | −0.2625 | 8.7 |

**512 is free** — AG-ARI −0.0019 is below every noise scale measured here and Foreground IoU
*improves*. It is also only 7% faster: the rollout is not the bottleneck (`PV2_0154`).

---

## 3. Best performance — model 253, trained and evaluated with CLEVR

Each row adds **1,000 unseen images × 10 epochs** (630 updates) to the finished core. Object-count
accuracy is the fraction of the 320 images whose predicted object count equals the truth; the
truth averages **6.36** objects per image.

| train data | FG-ARI | Foreground IoU | Object IoU | Δ FG-ARI | object count accuracy |
|---|---|---|---|---|---|
| 10,000 | 0.8421 | 0.7545 | 0.6410 | — | 0.175 |
| 11,000 | 0.8426 | 0.7539 | 0.6427 | +0.0005 | 0.203 |
| 12,000 | -1 | -1 | -1 | -1 | -1 |
| 13,000 | -1 | -1 | -1 | -1 | -1 |
| 14,000 | -1 | -1 | -1 | -1 | -1 |
| 15,000 | -1 | -1 | -1 | -1 | -1 |
| 16,000 | -1 | -1 | -1 | -1 | -1 |
| 17,000 | -1 | -1 | -1 | -1 | -1 |
| 18,000 | -1 | -1 | -1 | -1 | -1 |

**Status:** 12,000 is training now; 13,000–18,000 queued (`PV2_0155`, ~7 h per row). The first
increment moved FG-ARI by **+0.0005** — nothing. For scale, `cont5000` (5,000 extra images) moved
it **+0.0084**, so 1,000 may be below this curve's resolution.

> The graph of FG-ARI against train data is the deliverable here; with two points there is no
> curve yet.

---

## 4. Optimally trained comparison — each model at its own full recipe

**This table is almost entirely unmeasured, and that is the honest state.** Protocol B — every
model at its published budget on our render and our protocol — has not been run for any model
except Slot Attention's published checkpoint. The published FG-ARI figures these papers report
are on the **pixel protocol and their own test splits**, so they cannot be pasted into these
columns; they are given in the `trained info` column instead, as context, not as results.

| | CLEVR AG-ARI | CLEVR FG-IoU | CLEVR Obj-IoU | Tetro AG-ARI | Tetro FG-IoU | Tetro Obj-IoU | mdS AG-ARI | mdS FG-IoU | mdS Obj-IoU | trained info |
|---|---|---|---|---|---|---|---|---|---|---|
| **253** | **0.8421** | **0.7545** | **0.6410** | **0.8580** | **0.7885** | **0.8173** | **0.6974** | **0.8177** | **0.7628** | CLEVR 10,000 × 10 = 100,032 exposures. **This is our full recipe**, so rows 1 and 4 are the same for us. |
| Slot Attention | 0.3747 | 0.2219 | 0.1165 | -1 | -1 | -1 | -1 | -1 | -1 | published checkpoint, ~32M exposures (320×), trained on the **original CLEVR render**, not ours. Transfer, single checkpoint. |
| AKOrN (conv) | -1 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | published: 300 epochs × ~70k ≈ 21M exposures (210×). Published 3-seed FG-ARI (pixel protocol): Tetro 76.4±0.8, dSprites 63.8±7.7, CLEVR 59.0±4.3 |
| AKOrN (attn) | -1 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | same budget. Published 3-seed: 88.6±1.7 / 78.3±1.3 / 91.0±0.5 |
| AKOrN attn + up-tiling ×4 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | same budget, eval-time option. Published 3-seed: 93.1±0.3 / 87.1±1.0 / 94.6±0.7 |
| ItrSA | -1 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | same budget. Published 3-seed: 85.8±0.8 / 68.1±1.4 / 82.5±1.7 |
| CAE++ | -1 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | published: 100,000 steps × batch 64 = 6.4M exposures (64×). Published dSprites 0.51 — **with the per-image object-count oracle** |
| CtCAE | -1 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | same budget. Published dSprites 0.56, **oracle** |
| CAE | -1 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | same budget. Published dSprites 0.05, **oracle** |

### Published checkpoints: searched, and there are none

The cheap way to fill this table would be to take each author's released weights and run only our
evaluation on them. **That is not available for any model except Slot Attention.** Checked on
2026-11-10, three ways:

| repo | README weight links | GitHub releases | any weight URL in the tree |
|---|---|---|---|
| `autonomousvision/akorn` | none | **no releases** | only `mae_pretrain_vit_base.pth`, an ImageNet MAE backbone, not a segmentation checkpoint |
| `agopal42/ctcae` | none (data only, zenodo 4895643) | `public-init`, **zero assets** | none |
| `agopal42/syncx` | none (data only) | `public-init`, **zero assets** | none |
| `loeweX/RotatingFeatures` | none (data only) | **no releases** | none |
| `loeweX/ComplexAutoEncoder` | none (data only) | **no releases** | none |

Every download link in all five repositories points at a **dataset**, not at weights. So the
eight `-1` rows cannot be filled by evaluation alone — each one requires training the model at its
published budget ourselves, which is the cost below.

**What it would cost to fill.** AKOrN: ~3.5 h per model on our 10,000, or ~24 h on the full 70,000.
CAE family: 64× the runs already done, ~18 h per model. Slot Attention: its 500k-step recipe
(`PV2_0139`, written, unrun). Each also needs evaluating on all three datasets.

**Slot Attention is the only baseline whose authors released weights**, which is why it is the only
non-`-1` row here — and its Tetrominoes and Multi-dSprites cells are still `-1` because that
checkpoint has only been run on CLEVR. Those two are cheap (inference only) and are the single
easiest thing left in this table.

**One row already answers part of it.** At 1/64 of its budget CAE **exceeds** its published CLEVR
number by its own metric (0.1025 vs 0.04), and AKOrN (conv) at 1/210 **matches** its published
CLEVR number (60.5 vs 59.0±4.3). Those two are not starved, so our margins over them are not a
data-efficiency artefact. AKOrN (attn), ItrSA, CAE++, CtCAE and Slot Attention are starved, and
for those the margins are a data-efficiency claim (`PV2_0151`).

---

## Sources

| table | records |
|---|---|
| 1 | `PV2_0149`, `PV2_0150`, `PV2_0151` |
| 2 | `PV2_0154` |
| 3 | `PV2_0155` (running), `PV2_0157` for the counts |
| 4 | `PV2_0149` (Slot Attention published checkpoint); the rest unrun |
