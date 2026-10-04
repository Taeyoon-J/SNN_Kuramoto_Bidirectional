# Current status

- Goal: active. The three-seed mean of a spike or membrane readout must exceed the comparable Slot Attention mean on all three patch metrics.
- Baseline seed 0 checkpoint: `/Data0/kevinswk/patch_v2_sw/trained_models/baseline_best_seed0_20261003/core.pt`.
- SW_0001: completed classifier-only validation experiment. Fixed-k spike synchrony clustering improves all three metrics over the baseline spatial-components readout, but does not meet the Slot Attention reference.
- Gradient connectivity confirmed: phase-only loss gives no gradients to dendritic or membrane parameters (0/3 and 0/1 tensors connected). See `baseline_gradient_connectivity.json` on server.
- SW_0002: finished 10-epoch membrane-PLV pilot. Spike-synchrony FG-ARI rose to 0.270323, but foreground IoU and object IoU fell to 0.235452 and 0.123274 versus the 40-epoch starting checkpoint. No all-metric gain; training duration also differs. See SW_0002 report.
- SW_0003: matched 40-epoch membrane-PLV validation complete. FG-ARI improved to 0.2889, but foreground IoU and object IoU fell to 0.2075 and 0.1027. No all-metric gain.
- Graph diagnostic on validation IDs 1320–1351: mean same-object graph weight 0.254288 versus different-object 0.021720. SW_0004 graph-to-membrane loss pilot finished: spike FG-ARI 0.202940, foreground IoU 0.245920, matched object IoU 0.142077. Dendritic 3/3 and membrane 1/1 parameters have gradients, but there is no all-metric gain. Ten epochs versus 40 in baseline limits causal comparison.
- SW_0005: background-border rule screen completed. Best foreground IoU 0.3220 came with lower FG-ARI 0.0495 and object IoU 0.1386; not adopted.
- SW_0006: peer `patch_v2:5a29422` connected-components spike readout was re-evaluated on our fixed validation split. At correlation threshold 0.95, aggregate spike readout reached FG-ARI 0.077165, foreground IoU 0.328720, matched object IoU 0.173560. Both IoUs beat SW_0001, but FG-ARI fell from 0.136286. Prediction splits into about 14.88 foreground groups per image. Full threshold sweep in SW_0006/results.json; no production classifier merge yet.
- No training process has been started for SW_0001.
