# Experiment index

| id | status | change | fg_ari | fg_iou | obj_iou |
| --- | --- | --- | --- | --- | --- |
| PV2_0001 | completed | baseline under contract v1 | 0.4457 | **0.4115** | 0.3438 |
| PV2_0002a | failed | constrain the firing rate | 0.2316 | 0.3388 | 0.2306 |
| PV2_0002b | completed | geodesic coupling graph | **0.5537** | 0.3696 | **0.3809** |
| PV2_0003 | completed | per-component spikes and pulse coupling, screened at one seed | | | |
| PV2_0004 | completed | per-component spikes on the geodesic graph | 0.5105 | **0.4967** | **0.3689** |
| PV2_0005 | completed | ceiling re-measured; spatial prior 0.35 helps, geodesic 5 hurts | | | |
| PV2_0006 | completed | `--readout plv` added; the phase readout beats the spike readout by 0.174 | | | |
| PV2_0007 | failed | distil phase synchrony into the spikes by MSE; foreground IoU collapses to 0.37 | | | |
| PV2_0008 | completed | **spike synchrony in the objective, three seeds** | 0.6075 | **0.6755** | **0.4298** |
| PV2_0009 | completed | objective re-balance screened on seed 0; bimodality 3 lifts all three | | | |
| PV2_0010 | completed | bimodality 3 at three seeds: compresses spread, mean +0.022 | 0.6337 | 0.6643 | 0.4173 |
| PV2_0011 | refuted | fg_ari does not prefer a lower spike weight; 5 is still best | | | |
| PV2_0012 | completed | **half of a six-seed draw fails outright** | | | |
| PV2_0013 | completed | **bimodality 6 lifts all three at all three seeds** | **0.6756** | 0.6878 | **0.4728** |
| PV2_0014 | failed | align to the phase contrast; gap closes by the phase side collapsing | | | |
| PV2_0015 | completed | **the spiking path has 8 weights of transduction; no loss can fix the leak** | | | |

Three seeds on the test split, scored from spike masks. Peer experiments from
`patch_v2_sw` use the `SW_` prefix and are reviewed under `peer_updates/`.

## Reference

| | fg_ari | fg_iou | obj_iou |
| --- | --- | --- | --- |
| Slot Attention, ckpt-500, single published checkpoint | 0.6195 | 0.1241 | 0.0920 |

Not a 3-seed training mean, and trained on a different render; see
`baselines/slot_attention.md`.

## Where the goal stands

The goal is foreground IoU above 0.7. `PV2_0004` reaches **0.497**, up from the
baseline's 0.412, and is the first setting above baseline on all three metrics.

`PV2_0008` is the best result: test, three seeds, foreground IoU **0.6755**
(+0.1788 over `PV2_0004`, +36%), fg_ari 0.6075, matched-object IoU 0.4298. **The
goal is not met** -- 0.0245 short of 0.700 -- and fg_ari is 0.0120 below the Slot
Attention reference, so the all-three condition is not met either.

`PV2_0013` improves on it: raising the bimodality weight to 6 gives fg_ari
**0.6756**, foreground IoU 0.6878 and matched-object IoU 0.4728 on the full
validation split at three seeds, up on all three.

Two things block the goal. **The setting is unreliable** -- `PV2_0012` ran seeds 3,
4 and 5 of the same configuration and got foreground IoU 0.163, 0.430 and 0.000,
seed 5 having not trained at all, so `PV2_0008`'s test number is a lucky draw of
the three protocol seeds. And **the phase readout still beats the spike readout**,
0.7120 against 0.6756 on fg_ari.

`PV2_0015` explains why the second one cannot be fixed with a loss. The spiking
path holds 1,292 of 200,114 parameters, and the map from phase into it --
`oscillator_dense` -- is 8 weights shared across all 256 regions. Gradients reach
it and the alignment term will not move at any weight. The leak is an almost
parameterless transduction bottleneck, not a training failure, which is also why
`PV2_0007` and `PV2_0014` failed. Closing it needs a core change: give that
transduction per-region capacity.

The graph is spent: `PV2_0005` found a ground-truth coupling graph worth only
about +0.06. `PV2_0006` located the real gap by adding `--readout plv`: with one
classifier and one evaluation, the phase readout reaches 0.660 where the spike
readout reaches 0.486, so theta already carries what the goal needs and the
spiking path loses 0.174 of it. `PV2_0007` tried to close that by MSE-matching the
phase matrix and failed badly -- matching values fuses the graph; what the readout
needs is the contrast.

## Diagnostics kept out of the headline

Oracle results, per the contract. Never compared against Slot Attention.

| | fg_ari | fg_iou | obj_iou |
| --- | --- | --- | --- |
| trained on a ground-truth coupling graph | 0.5370 | 0.6096 | 0.3884 |
| phase/PLV readouts, every variant | — | — | — |
