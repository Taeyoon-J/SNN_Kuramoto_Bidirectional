# SW0057 fixed multi-readout summary

Fixed validation IDs 1320-1639; long T1024/settle512; seeds 0/1/2.
Ground truth is used only for scoring. Mean and sample standard deviation are across seeds.

| Readout | Metric | Mean | Sample SD | Seed 0 | Seed 1 | Seed 2 |
|---|---|---:|---:|---:|---:|---:|
| spike_cc_threshold_0p50 | fg_ari | 0.626965 | 0.184685 | 0.817431 | 0.614801 | 0.448663 |
| spike_cc_threshold_0p50 | foreground_iou | 0.423564 | 0.274809 | 0.730732 | 0.338944 | 0.201018 |
| spike_cc_threshold_0p50 | matched_object_iou | 0.440859 | 0.210961 | 0.644500 | 0.454805 | 0.223270 |
| membrane_spatial_sigma1p5_k10 | fg_ari | 0.738478 | 0.081293 | 0.832345 | 0.691080 | 0.692009 |
| membrane_spatial_sigma1p5_k10 | foreground_iou | 0.222631 | 0.020028 | 0.237224 | 0.230871 | 0.199798 |
| membrane_spatial_sigma1p5_k10 | matched_object_iou | 0.368848 | 0.125485 | 0.486377 | 0.383476 | 0.236690 |
