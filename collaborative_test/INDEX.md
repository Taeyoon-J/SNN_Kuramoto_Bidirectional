# Collaborative experiments

| ID | Status | Change | Validation result |
|---|---|---|---|
| SW_0001 | completed | Spike synchrony spectral readout, fixed k=8 | FG-ARI 0.136286; foreground IoU 0.271997; matched object IoU 0.145604 on CLEVR IDs 1320–1639 |
| SW_0002 | complete, mixed result | Membrane-sourced synchrony objective, 10-epoch pilot | Spike FG-ARI 0.270323, foreground IoU 0.235452, object IoU 0.123274 on validation; not an all-metric gain |
| SW_0003 | complete, mixed result | Matched 40-epoch membrane-PLV follow-up to isolate source change | FG-ARI 0.2889, foreground IoU 0.2075, object IoU 0.1027; no all-metric gain |
| SW_0004 | complete, mixed result | Preserve phase PLV; add graph-to-membrane synchrony KL at weight 0.1, 10-epoch pilot | Spike FG-ARI 0.202940; foreground IoU 0.245920; matched object IoU 0.142077. Membrane gradients connected, but no all-metric gain |
| SW_0005 | complete, mixed result | Prediction-only perimeter rule for background clusters; no model retraining | Foreground IoU rose to 0.3220, but FG-ARI fell to 0.0495 and object IoU to 0.1386 |
| SW_0006 | complete, mixed result | Reproduce peer `patch_v2:5a29422` connected-components spike classifier on our fixed HDF5 split | At threshold 0.95 aggregate: FG-ARI 0.077165, foreground IoU 0.328720, matched object IoU 0.173560; better IoUs but worse ARI than SW_0001 |
| SW_0007 | complete, negative result | Peer connected-component foreground selection followed by spike-pattern spectral grouping of only foreground patches | Threshold 0.95, k=5: FG-ARI 0.080966; foreground IoU 0.328720; matched object IoU 0.134053. No all-metric gain |
| SW_0008 | complete, improvement candidate | Apply peer spike classifier to SW_0003 and SW_0004 trained cores on same validation split | SW_0004 + component-product threshold 0.50: FG-ARI 0.195269; foreground IoU 0.275711; matched object IoU 0.238803, all above SW_0001 seed-0 validation |
| SW_0009 | complete, goal unmet | Frozen SW_0008 readout on reference test IDs 1000–1319 | FG-ARI 0.190909; foreground IoU 0.270928; matched object IoU 0.229805. Slot reference 0.890115 / 0.212251 / 0.235487 |
| SW_0010 | complete, negative result | Lower peer component-product spike edge threshold to reduce object fragmentation; validation only | At 0.10 groups fall 30.99→18.77, but FG-ARI 0.183586, foreground IoU 0.266737, object IoU 0.197857 all below threshold 0.50 |
| SW_0011 | training | Match the SW_0004 graph-teacher pilot to 40 training epochs, seed 0 | Awaiting validation |
| SW_0012 | complete, diagnostic only | Test peer `patch_v2:0dd2115` geodesic graph distance on our existing checkpoint, without modifying core | Same/different-object edge ratio 15.00→13.59 on 16 validation images; not merged |

Each experiment's README begins with a Korean plain-language explanation and a
baseline-versus-candidate table when results exist. Arrows in those tables
mean higher is better; no score is combined into a single total. These are
validation scores from seed 0, not three-seed test results or a goal claim.
