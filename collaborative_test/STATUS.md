# Current status

- Goal: active. The three-seed mean of a spike or membrane readout must exceed the comparable Slot Attention mean on all three patch metrics.
- Baseline seed 0 checkpoint: `/Data0/kevinswk/patch_v2_sw/trained_models/baseline_best_seed0_20261003/core.pt`.
- SW_0001: completed classifier-only validation experiment. Fixed-k spike synchrony clustering improves all three metrics over the baseline spatial-components readout, but does not meet the Slot Attention reference.
- Gradient connectivity confirmed: phase-only loss gives no gradients to dendritic or membrane parameters (0/3 and 0/1 tensors connected). See `baseline_gradient_connectivity.json` on server.
- SW_0002: finished 10-epoch membrane-PLV pilot. Spike-synchrony FG-ARI rose to 0.270323, but foreground IoU and object IoU fell to 0.235452 and 0.123274 versus the 40-epoch starting checkpoint. No all-metric gain; training duration also differs. See SW_0002 report.
- No training process has been started for SW_0001.
