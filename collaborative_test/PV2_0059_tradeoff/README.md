# PV2_0059 — the fg_ari / IoU dial, swept on checkpoints we already had

**completed.** Parent `PV2_0058`. Evaluation only: no training, no weights changed,
seed 0, validation 300, contract v1.

## Why

`PV2_0058` showed fg_ari follows predicted foreground extent: chroma3 reached 0.7380
while predicting 0.509 foreground against a target of 0.118, with foreground IoU down
at 0.1968. `PV2_0049` had measured the same shape in the baseline — Slot predicts
0.85 and gets fg_ari 0.6635 with fg_iou 0.1184. Our models sit at 0.129, nearly
calibrated, which is where the 6x IoU advantage comes from.

That raises a question never swept: **is there a readout setting where fg_ari clears
0.75 while both IoUs stay well above Slot's?** `PV2_0056` had already seen the readout
family has different trade-offs — spectral k=10 on these checkpoints gives
0.7126 / 0.4935 / **0.5720** against connected components' 0.7239 / 0.7251 / 0.4951,
lower foreground IoU but *higher* matched-object IoU — but both were single points,
neither chosen by a sweep.

22 settings across three axes, all on `BIM6_s1`, after a regression gate reproducing
0.7239 / 0.7251 / 0.4951.

## Result

Spatial kernel width and the membrane threshold barely move anything — sigma 0.5 to
3.0 spans fg_ari 0.7142 to 0.7272, and the population threshold k from -1.0 to 0.5
spans 0.6847 to 0.7261, with predicted foreground pinned near 0.128 throughout.

**Spectral k is the dial.**

| setting | fg_ari | fg_iou | obj_iou | groups | pred_fg |
| --- | --- | --- | --- | --- | --- |
| connected components | 0.7239 | **0.7251** | 0.4951 | 5.46 | 0.129 |
| spectral k=6 | 0.5789 | 0.5416 | 0.3857 | 4.84 | 0.207 |
| spectral k=10 | 0.7126 | 0.4935 | 0.5720 | 9.00 | 0.355 |
| spectral k=12 | 0.7364 | 0.4253 | 0.6216 | 11.00 | 0.452 |
| **spectral k=14** | **0.7562** | 0.3266 | 0.6551 | 13.00 | 0.573 |
| spectral k=16 | **0.7672** | 0.2396 | 0.6705 | 15.00 | 0.683 |
| spectral k=20 | **0.7720** | 0.1561 | **0.6944** | 19.00 | 0.827 |

Four points clear fg_ari 0.75, and all four still beat Slot Attention on both IoUs.
fg_ari and matched-object IoU rise with k while foreground IoU falls, monotonically,
with predicted foreground tracking k from 0.21 to 0.83.

**Spectral k is a plain hyperparameter** — a fixed group count, never the true object
count (that is `--spectral-oracle-k`, which is not used here) — so this is not an
oracle readout, and the contract allows selecting it on validation.

## How to read it

This is a dial between our calibrated-foreground regime and Slot Attention's
over-predicting one, not free improvement. k=20 predicts 0.827 foreground, which is
Slot's own pathology, so there is no reason to walk further along it. The honest
report is the whole curve with what each point costs, which `PV2_0060` then measured
at three seeds.
