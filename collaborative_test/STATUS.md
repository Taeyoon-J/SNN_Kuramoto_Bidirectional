# Status

Contract v1, split manifest v1, targets built. Classifier rebuilt for the
contract. No experiment has a 3-seed number on test yet.

## Where things stand

| | |
| --- | --- |
| best measured | `D055_s2`, validation, one seed: fg_ari 0.528, fg_iou 0.421, obj_iou 0.413 |
| reference | Slot Attention ckpt-500 on test: fg_ari 0.620, fg_iou 0.124, obj_iou 0.092 |
| seed spread | 0.13 on fg_ari between two seeds of the same config, which is wider than most differences measured so far |

`D055` is the configuration in `snn_kuramoto_bidirectional/configs/best_16x16.sh`:
16x16, top_k 32, k 256, freq_gain 2.0, gate_mode raw, graph_spatial_decay 0.55,
40 epochs. No pulse coupling, no per-component spiking.

## Next

1. `PV2_0001` -- 3-seed baseline on test with the chosen threshold. The D055
   seeds are already trained; this is evaluation only.
2. `PV2_0002` -- the coupling graph. It is the one measured lever: training with
   a ground-truth graph (a diagnostic, not a result) gives fg_iou 0.610 against
   0.421, and lifts the phase readout from 0.598 to 0.723.

## Open question for the user

The Slot Attention reference is one published checkpoint trained on the original
CLEVR render, evaluated here on `clevr_with_masks`, a different render. Its own
authors say the two are not directly comparable. It loses fg_iou and obj_iou
badly because it splits the background across slots, which is what an
out-of-distribution render would do. Either report it with that stated, or
retrain Slot Attention on this split -- the second is fair and costs a TF
training pipeline and many hours.

## Resume

Everything is under `/work/USERS/tkim1` on the UNC cluster; code in
`/export_home/tkim1`. The SSH ControlMaster expires -- `ssh frontier` once
re-opens it.
