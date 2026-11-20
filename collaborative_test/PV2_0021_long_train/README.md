# PV2_0021-0022 — training at the readout's window fails; bimodality 6 holds

**failed / exhausted.** Parent `PV2_0020`. Seeds 0/1/2, validation 300, readout
window 1024.

## Training at a longer window makes it much worse

If the readout needs 1024 and training optimises synchrony over 32 settled steps,
the obvious move is to match them. It is wrong.

| | spike fg_ari | phase fg_ari |
| --- | --- | --- |
| trained at window 64 | **0.6887** | **0.7110** |
| trained at window 256 | 0.5596 | 0.5654 |

Seed 1 broke outright: foreground IoU 0.178 with predicted foreground 0.50.

This is consistent with something already in this repository -- the rollout is
chaotic and the readout is not. Backpropagating through 256 steps feeds that
chaos into the gradient. **Short-window training with a long-window readout is
the right split**, and the existing configuration already had it.

## Bimodality 6 survives re-selection at the correct window

The weight was chosen while the evaluator still defaulted to a 256-step readout,
so it was re-selected at 1024:

| weight | 4 | **6** | 8 |
| --- | --- | --- | --- |
| fg_ari | 0.6199 | **0.6887** | 0.6351 |

6 remains the peak. Seed 0 breaks at every weight -- fg_ari 0.505 at 4, foreground
IoU 0.412 with predicted foreground 0.258 at 8 -- which is the membrane threshold
bug recorded in `PV2_0023`, not something an objective weight can reach.
