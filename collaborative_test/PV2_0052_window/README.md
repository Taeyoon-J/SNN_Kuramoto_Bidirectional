# PV2_0052 — the integration window is already saturated

**completed. Negative result, and it corrects an expectation I had carried from
`PV2_0019`.** Parent `PV2_0044`. Evaluation only — no training, no new weights.
Validation 300, contract v1, sync 0.05.

## Why it was tried

`PV2_0019` established that the phase-to-spike loss is a **sampling** limit rather
than a parameterisation problem: as the window grew 64 -> 1024 the spike readout
went 0.512 -> 0.688 while the phase readout saturated at 0.711 by window 128, and
the gap between them fell 0.149 -> 0.023 monotonically. Spike correlations are
estimated from binary events, so their variance falls with the number of samples
and nothing has to change for the estimate to improve.

Every number in this project, 0.7250 included, was read at window 1024 — simply
where that sweep stopped. The curve had not flattened and 0.023 was still
outstanding, so 2048 and 4096 were the untested continuation of a measured trend.

## Result

| window | fg_ari | fg_iou | obj_iou | fg_ari std | vs 1024 |
| --- | --- | --- | --- | --- | --- |
| 1024 | 0.7250 | 0.7188 | 0.4953 | 0.0022 | — |
| 2048 | 0.7251 | 0.7193 | 0.4954 | 0.0010 | **+0.0002** |
| 4096 | **0.7254** | 0.7205 | 0.4958 | 0.0020 | **+0.0005** |

Per checkpoint: FZG_s0 0.7235 / 0.7263 / 0.7253, BIM6_s1 0.7239 / 0.7244 / 0.7235,
FZG_s2 0.7275 / 0.7247 / 0.7275.

**+0.0005 at four times the window**, inside the seed spread of 0.0020. The axis is
closed.

## Why the expectation was wrong

`PV2_0019`'s sweep was run on **pre-graph-freeze checkpoints**. These checkpoints
are already at their sampling limit at window 1024, so the 0.023 that was still
outstanding there had already been recovered here. I carried a measurement from one
set of checkpoints onto another without checking that it transferred.

## What it does establish

**0.7250 is not an artifact of the window.** Quadrupling the integration time moves
it by 0.0005, so the readout is stable in the regime it is being reported in.
