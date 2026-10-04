# Collaborative experiments

| ID | Status | Change | Validation result |
|---|---|---|---|
| SW_0001 | completed | Spike synchrony spectral readout, fixed k=8 | FG-ARI 0.136286; foreground IoU 0.271997; matched object IoU 0.145604 on CLEVR IDs 1320–1639 |
| SW_0002 | complete, mixed result | Membrane-sourced synchrony objective, 10-epoch pilot | Spike FG-ARI 0.270323, foreground IoU 0.235452, object IoU 0.123274 on validation; not an all-metric gain |
| SW_0003 | complete, mixed result | Matched 40-epoch membrane-PLV follow-up to isolate source change | FG-ARI 0.2889, foreground IoU 0.2075, object IoU 0.1027; no all-metric gain |
| SW_0004 | complete, mixed result | Preserve phase PLV; add graph-to-membrane synchrony KL at weight 0.1, 10-epoch pilot | Spike FG-ARI 0.202940; foreground IoU 0.245920; matched object IoU 0.142077. Membrane gradients connected, but no all-metric gain |
| SW_0005 | complete, mixed result | Prediction-only perimeter rule for background clusters; no model retraining | Foreground IoU rose to 0.3220, but FG-ARI fell to 0.0495 and object IoU to 0.1386 |
| SW_0006 | complete, mixed result | Reproduce peer `patch_v2:5a29422` connected-components spike classifier on our fixed HDF5 split | At threshold 0.95 aggregate: FG-ARI 0.077165, foreground IoU 0.328720, matched object IoU 0.173560; better IoUs but worse ARI than SW_0001 |

Each experiment's README begins with a Korean plain-language explanation and a
baseline-versus-candidate table when results exist. Arrows in those tables
mean higher is better; no score is combined into a single total. These are
validation scores from seed 0, not three-seed test results or a goal claim.
