# PV2_0146 — the Tetrominoes head gain was a lucky draw; CLEVR's and Multi-dSprites' are not

**Status** completed. 5 head seeds × 3 datasets, 5 s each, no rented GPU.
**Corrects** the Tetrominoes head row reported in PV2_0134, PV2_0136 and BENCHMARK_PLAN_v4.

## Why this was worth doing for free

Three-seed **core** means need rented GPUs — 51–64 h per seed on frontier against 18.5 h on a
rented RTX 4090. But the head trains in **5 seconds**. Holding the core fixed and redrawing only
the head's initialisation answers a narrower question that was still open: **is the reported head
gain reproducible, or was it one lucky draw?**

This is **not** a 3-seed core mean and does not replace one. It isolates head-training variance.

## Result

Arm `local_rgb_trust`, 2 epochs (the fixed label-free recipe), 5 head seeds, CLEVR-trained core
held fixed, same 320 evaluation images.

| dataset | metric | core | gain, mean over 5 head seeds | head-seed std | verdict |
|---|---|---|---|---|---|
| **CLEVR** | fg_ari | 0.8421 | **+0.0047** | 0.0004 | **clears 2σ by 11×** |
| | foreground_iou | 0.7545 | **+0.0030** | 0.0003 | **clears, 10×** |
| | matched_object_iou | 0.6410 | **+0.0050** | 0.0005 | **clears, 10×** |
| **Multi-dSprites** | fg_ari | 0.6974 | **+0.0064** | 0.0011 | **clears, 6×** |
| | foreground_iou | 0.8177 | **+0.0057** | 0.0027 | **clears** |
| | matched_object_iou | 0.7628 | **+0.0059** | 0.0023 | **clears** |
| **Tetrominoes** | fg_ari | 0.8580 | **−0.0020** | 0.0033 | **inside noise, sign negative** |
| | foreground_iou | 0.7885 | +0.0003 | 0.0005 | inside noise |
| | matched_object_iou | 0.8173 | **−0.0008** | 0.0016 | **inside noise, sign negative** |

## The correction

**The Tetrominoes head gain does not exist.** It was reported as **+0.0006 / +0.0010 / +0.0015**
in PV2_0134, PV2_0136 and `BENCHMARK_PLAN_v4.md`. Those are head **seed 0's** numbers. Averaged
over five head seeds the gain is **−0.0020 / +0.0003 / −0.0008** — negative on two of the three
metrics and inside the head-seed spread on all three.

PV2_0141 had already flagged these three cells as inside the epoch-and-arm wobble, and PV2_0144
put the core-seed std at 0.0396 there. This settles it from a third direction, the cheapest one:
even holding the core fixed, redrawing the head alone moves the score by more than the claimed
gain.

**CLEVR and Multi-dSprites survive this test cleanly.** CLEVR's std is one eleventh of its gain.
So "it was a lucky head draw" is ruled out for those two.

## What "all nine cells improve" has become

| claim | status |
|---|---|
| all 9 cells improve (PV2_0136, as written) | **withdrawn** |
| 4 of 9 clear the epoch/arm wobble (PV2_0141) | stands |
| **6 of 9 are stable under head re-initialisation** — CLEVR 3, Multi-dSprites 3 | **this test** |
| Tetrominoes 3 of 3 | **not real** |
| any of it as a 3-seed core mean | **still missing** |

The head gain is real and reproducible **on CLEVR and Multi-dSprites, given this core**. Whether
it survives a *different* core is the question only rented-GPU core seeds can answer.

## Reproduce

```bash
python headseed.py --code benchmark_codes --config configs/frontier.json --run <run> \
  --dataset clevr --train-cache ev/clevr_train --eval-cache ev/clevr_fixed320 \
  --eval-split fixed320 --arm local_rgb_trust --epochs 2 --seeds 5 --out headseed_clevr.json
```
