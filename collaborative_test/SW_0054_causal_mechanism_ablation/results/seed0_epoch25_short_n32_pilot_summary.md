# SW0054 count-32 pilot summary

Checkpoint SHA-256: `d30147fbd460a702689cb59fea13dd4d36d70b4a7b460ffd020382b2334e8c85`
Protocol: IDs 1320-1351, T256/settle64, pilot condition set, GT-free prediction.

Deltas are intervention minus normal. Activity values are ratios to normal.

| Condition | Readout | FG-ARI delta | FG IoU delta | Matched IoU delta | Spike-rate ratio | Membrane-variance ratio | Membrane-abs ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| gate_perm_s0 | spike_cc | -0.674001 | -0.603998 | -0.462112 | 1.0005366608234825 | 0.9989897642671285 | 1.0005336768039321 |
| gate_perm_s0 | membrane_spatial | -0.177434 | 0.007768 | -0.152926 | 1.0005366608234825 | 0.9989897642671285 | 1.0005336768039321 |
| carrier_perm_s0 | spike_cc | -0.004832 | -0.012223 | -0.007555 | 1.0005622390821556 | 1.0006399194047328 | 1.0006340722493436 |
| carrier_perm_s0 | membrane_spatial | -0.064457 | 0.006144 | -0.061322 | 1.0005622390821556 | 1.0006399194047328 | 1.0006340722493436 |
| gate_mean | spike_cc | -0.666685 | -0.563001 | -0.477229 | 1.011894372891679 | 0.30244141120499757 | 0.9809648732241784 |
| gate_mean | membrane_spatial | -0.128889 | 0.003613 | -0.084994 | 1.011894372891679 | 0.30244141120499757 | 0.9809648732241784 |
| carrier_mean | spike_cc | 0.005710 | -0.001696 | -0.006463 | 1.0120255217934606 | 0.5076867052549898 | 1.0145205274168942 |
| carrier_mean | membrane_spatial | -0.008959 | 0.003037 | 0.014262 | 1.0120255217934606 | 0.5076867052549898 | 1.0145205274168942 |
| K0 | spike_cc | -0.430772 | -0.253211 | -0.268714 | 1.0007519042832604 | 0.9781249807030264 | 0.9929789624148877 |
| K0 | membrane_spatial | -0.103416 | 0.005164 | -0.124597 | 1.0007519042832604 | 0.9781249807030264 | 0.9929789624148877 |

## Distance-controlled AUC delta

| Condition | Phase | Gate | Carrier | H-wave | Membrane | Spike |
|---|---:|---:|---:|---:|---:|---:|
| gate_perm_s0 | 0.000000 | -0.375544 | 0.000000 | -0.223031 | -0.213625 | -0.337514 |
| carrier_perm_s0 | 0.000000 | 0.000000 | -0.380815 | -0.082187 | -0.080955 | -0.009273 |
| gate_mean | 0.000000 | -0.391347 | 0.000000 | 0.002204 | -0.003760 | -0.372264 |
| carrier_mean | 0.000000 | 0.000000 | -0.386366 | -0.007758 | -0.007699 | 0.019083 |
| K0 | -0.076230 | -0.179480 | -0.269359 | -0.228041 | -0.227170 | -0.163979 |

Numerical deltas only; this summary does not infer whether an effect is explained by activity-scale collapse.
