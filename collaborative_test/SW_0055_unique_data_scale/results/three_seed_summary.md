# SW0055 three-seed summary

Validation IDs 1320-1639 (320 images). Primary endpoint: long T1024/settle512 at fixed threshold .50; .35 is secondary. No posthoc threshold selection.

| Window | Threshold | Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Population SD | SW0053 baseline | Delta vs baseline |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| long | 0.50 | fg_ari | 0.817116 | 0.614845 | 0.448238 | 0.626733 | 0.150828 | 0.562902 | +0.063831 |
| long | 0.50 | foreground_iou | 0.730071 | 0.338969 | 0.200981 | 0.423340 | 0.224088 | 0.381328 | +0.042012 |
| long | 0.50 | matched_object_iou | 0.643890 | 0.454188 | 0.222280 | 0.440119 | 0.172409 | 0.356857 | +0.083262 |
| long | 0.35 | fg_ari | 0.810630 | 0.607738 | 0.450793 | 0.623054 | 0.147302 | 0.560105 | +0.062949 |
| long | 0.35 | foreground_iou | 0.730350 | 0.335704 | 0.201251 | 0.422435 | 0.224541 | 0.381852 | +0.040583 |
| long | 0.35 | matched_object_iou | 0.633323 | 0.441718 | 0.220506 | 0.431849 | 0.168676 | 0.349261 | +0.082588 |
| short | 0.50 | fg_ari | 0.765027 | 0.589180 | 0.435789 | 0.596665 | 0.134515 | n/a | n/a |
| short | 0.50 | foreground_iou | 0.769182 | 0.384011 | 0.198056 | 0.450416 | 0.237842 | n/a | n/a |
| short | 0.50 | matched_object_iou | 0.554573 | 0.408971 | 0.189466 | 0.384337 | 0.150069 | n/a | n/a |
| short | 0.35 | fg_ari | 0.740321 | 0.560759 | 0.432097 | 0.577726 | 0.126402 | n/a | n/a |
| short | 0.35 | foreground_iou | 0.771371 | 0.385611 | 0.204249 | 0.453743 | 0.236486 | n/a | n/a |
| short | 0.35 | matched_object_iou | 0.522353 | 0.375422 | 0.177882 | 0.358552 | 0.141135 | n/a | n/a |

Code SHA-256: `7a9f59b4be5ecce1024c5e127b43bd3bdc510797a619e2728be2e2bed176805f`  
Training gamma SHA-256: `7139439232d5717b66ffd1749572ab3faff9d0337d49c28b1602dce4d6c742aa`
