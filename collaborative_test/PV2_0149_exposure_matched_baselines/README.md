# PV2_0149 — Slot Attention trained on our exact data budget

**Status** Slot Attention done; three other baselines cloned but not yet trained.
**Protocol A** every model gets **100,000 image exposures** — the same budget our 253 core had
(10,000 unique × 10 epochs) — on the **same 10,000 CLEVR train ids**, scored by the same
`evaluate_patch_masks` on the same 320 fixed320 images.

## Result

Slot Attention: **1,563 steps × batch 64 = 100,032 exposures (10.0 epochs)**, 20 min on a
TITAN RTX. Its background rule was chosen on validation out of the same four fixed in advance.

| CLEVR fixed320 | training budget | fg_ari | foreground_iou | matched_object_iou |
|---|---|---|---|---|
| **253 core + label-free head (ours)** | **100,032 exposures** | **0.8468** | **0.7574** | **0.6456** |
| **Slot Attention, exposure-matched** | **100,032 exposures** | **0.2872** | 0.2220 | 0.0801 |
| Slot Attention, published checkpoint | ~32M exposures (**320×**) | 0.3747 | 0.2219 | 0.1165 |

**On an identical data budget we are 2.9× its FG-ARI and 8× its matched-object IoU.**

## The finding that matters more

> **QUALIFIED (2026-11-10).** What follows was measured on **CLEVR only**, and it does not hold
> for transfer. Run zero-shot on the other two datasets, the published checkpoint beats the
> exposure-matched one by a wide margin: Tetrominoes FG-ARI **0.4742 vs 0.1739 (2.7×)** and
> Multi-dSprites **0.3205 vs 0.1930 (1.7×)**. So the extra 320× of data buys +0.0875 on the
> render it trained on and **+0.3003 off it**. The sentence below is true of CLEVR and false as a
> general statement about the budget. Full rows in `RESULTS_TABLES.md`.

**Giving Slot Attention 320× more data barely moves it on our render**: FG-ARI 0.2872 → 0.3747,
and Foreground IoU is *identical* to four decimals (0.2220 → 0.2219). Both checkpoints use
**6.89 and 6.91 of 7 slots per image** with near-equal shares:

```
exposure-matched : 0:0.164 1:0.149 2:0.142 3:0.153 4:0.120 5:0.137 6:0.135
published        : 0:0.156 1:0.147 2:0.141 3:0.148 4:0.133 5:0.131 6:0.144
```

So Slot Attention **tiles rather than segments on these images regardless of how long it
trains**, which is why all four background rules give the identical FG-ARI (0.2899 validation,
0.2872 fixed320) — they only relabel patches the truth calls background, and FG-ARI excludes
those. The published checkpoint's weak score was never mainly a transfer artefact; training it
here from scratch reproduces the same behaviour.

## What this licenses, and what it does not

**Licenses:** "on an identical data budget, and on the same images and metric, our core reaches
0.8468 FG-ARI where Slot Attention reaches 0.2872." That is a **data-efficiency** claim and it
is now measured, not asserted.

**Does not license** "we beat Slot Attention." Two gaps remain:
- this is **seed 0** on both sides;
- Slot Attention at its **published 500,000-step budget on our render** has not been trained
  (`PV2_0139`). 1,563 steps is 1/320 of its recipe, and a model given 1/320 of its training is
  not the model its authors report.

Protocol B — each model at its own published recipe — is still the only version that supports
the stronger sentence. See `PV2_0145` for the three senses of beating Slot Attention.

## The other three baselines

Cloned and probed, none trained:

| repo | spec | state |
|---|---|---|
| `autonomousvision/akorn` | `requirements.txt` (pip) | **most promising** — Kuramoto-based, so the most direct competitor, and its published CLEVRTex FG-ARI 88.5 gives a way to verify the environment is correct |
| `loeweX/RotatingFeatures` | `environment.yml` (conda) | no entry point at the repo root; code under `codebase/` |
| `loeweX/ComplexAutoEncoder` | `environment.yml` + `setup.sh` | same |

Each needs its own environment, and none reads our `rgb128.npy` layout, so none can be queued
blind — the data loader has to be wired to our splits and the exposure budget converted to that
model's batch size. Rotating Features and the complex autoencoders are lower value to train
ourselves because their **published** numbers are already directly comparable on Multi-dSprites,
where our grid-64 protocol coincides with their pixel protocol (`PV2_0145`).

## Reproduce

```bash
python benchmark_codes/sa_dump_train.py --split train --out sa/frames_train.npy   # torch env
python benchmark_codes/sa_train.py --frames sa/frames_train.npy --steps 1563 \
  --batch 64 --seed 0 --out sa/exposure_matched_seed0                             # tf env
python benchmark_codes/sa_infer.py --checkpoint-dir sa/exposure_matched_seed0 ...  # tf env
python benchmark_codes/sa_bg.py --val ... --eval ... --out sa/A_bg_rules.json      # torch env
```
