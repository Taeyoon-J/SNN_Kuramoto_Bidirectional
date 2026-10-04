# SW_0002 — membrane-sourced synchrony pilot

Status: training and validation complete. This pilot does not meet the goal.

## Reason

Under the starting `plv_source=phase` objective, all PLV terms read Kuramoto
theta directly. A gradient-connectivity check on four training examples showed
zero connected gradient tensors in `dendric_layer` (0/3) and `membrane_layer`
(0/1), despite nonzero gradients in gamma projection, graph, and Kuramoto.
Thus the phase loss was not training the spike path. This is a causal property
of the current computation graph, not an inference from spike rate alone.

## Controlled change

Train seed 0 for 10 epochs with baseline model settings fixed except
`--plv-source membrane` instead of `--plv-source phase`. The starting baseline
checkpoint ran 40 epochs, so a direct score difference is confounded by training
duration; this is a short pilot, not a matched-epoch causal ablation. It measures centered
membrane-trace synchrony (not spike threshold events), connecting the objective
to the SNN layers. The unchanged `--spike-per-component` still produces the
final spike history. This pilot is not a replacement for the 40-epoch baseline.

The exact command is in `run.sh`. The server checkpoint is
`/Data0/kevinswk/patch_v2_sw/trained_models/SW_0002_membrane_plv_pilot_seed0/core.pt`.
The training objective fell from 3.495306 to 2.764854; that alone is not an
object-grouping result.

## Evaluation

`evaluate.sh` evaluated the original seed-0 checkpoint and this pilot on
exactly the same 320 validation images (IDs 1320–1639), with both the original
spatial-components readout and the fixed-k=8 aggregate-spike-synchrony readout
from SW_0001. The metric calculation uses the HDF5 instance mask converted to
16×16 modal-ID patches; no true object count is used in clustering.

| Seed-0 readout, validation 320 | FG-ARI | Foreground IoU | Matched object IoU | Actual spike rate |
|---|---:|---:|---:|---:|
| Baseline 40ep, spatial components | 0.000000 | 0.216821 | 0.014756 | 0.484850 |
| Pilot 10ep, spatial components | 0.000000 | 0.216821 | 0.014756 | 0.448761 |
| Baseline 40ep, spike synchrony k=8 | 0.136286 | 0.271997 | 0.145604 | 0.484850 |
| Pilot 10ep, spike synchrony k=8 | 0.270323 | 0.235452 | 0.123274 | 0.448761 |

The spike readout improves foreground ARI by 0.134037 but worsens foreground
IoU by 0.036545 and matched object IoU by 0.022331 relative to the starting
checkpoint. This is a tradeoff, not an all-metric improvement. The original
spatial-components classifier still assigns the entire image to one group.
No test-split metrics were used to select this pilot.

The full per-image results and patch-label predictions are on the server at
`$BASE/validation_fixed_split/{summary.json,patch_masks.pt}` and
`$PILOT/validation_fixed_split/{summary.json,patch_masks.pt}` (paths defined
in `evaluate.sh`).
