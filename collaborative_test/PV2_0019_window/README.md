# PV2_0019-0020 — the leak is an estimation limit, not a parameterisation one

**completed. This explains five earlier failures.** Parent `PV2_0013`. BIM6 seeds
0/1/2, validation 300. Nothing retrained, nothing re-parameterised: only the
length of the window the synchrony is estimated over.

| window | 64 | 128 | 256 | 512 | 1024 | 2048 |
| --- | --- | --- | --- | --- | --- | --- |
| spike fg_ari | 0.512 | 0.597 | 0.651 | 0.679 | **0.688** | flat |
| phase fg_ari | 0.661 | 0.702 | 0.709 | 0.710 | 0.711 | -- |
| gap | 0.149 | 0.105 | 0.058 | 0.031 | 0.023 | -- |

**The phase readout saturates by 128; the spike readout needs 1024.** The gap
falls monotonically with window length and the two never diverge again.

Spikes sample a continuous phase relationship as sparse binary events -- at a
12-14% firing rate a short window leaves a handful per patch -- and the
per-component product multiplies that sampling noise four times. The phase PLV
reads continuous phases and is well estimated from a fraction of the samples.

This is why `PV2_0007` (MSE onto the phase matrix), `PV2_0014` (row-standardised
contrast), `PV2_0015` (frozen phase path), `PV2_0016` (per-region capacity) and
`PV2_0018` (tied init) all failed. Every one of them tried to fix the shape of
the spike synchrony matrix. The problem was how few samples it was estimated
from.

For the research question: **binding-by-synchrony survives the oscillator
dynamics and reaches the spikes; reading it out of spikes costs 8 to 16 times the
window the phases need.**

## A measurement history correction

`evaluate_model.py` defaults to a 256-step window with 64 settled, while training
runs at 64 with 32. Every BIM6 number reported before this experiment was
therefore measured at 256, not at the training window of 64 -- including the
0.6887 that was called the best result. The numbers stand; the description of
them as "the training window" did not.

## Threshold, also exhausted

At window 1024 the synchrony threshold is flat across 0.05-0.20: 0.6877 / 0.6887 /
0.6871 / 0.6880. Nothing left on that axis.
