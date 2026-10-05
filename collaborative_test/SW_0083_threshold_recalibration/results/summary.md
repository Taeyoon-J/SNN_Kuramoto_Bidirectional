# SW0083 membrane-threshold recalibration

| vth | Event rate | Always-on | Constant histories | FG-ARI | FG IoU | Object IoU | All metrics improve |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.06 | 1.000000 | 1.000000 | 1.000000 | 0.708757 | 0.468105 | 0.509881 | no |
| 0.5 | 0.938791 | 0.701843 | 0.703094 | 0.595236 | 0.330631 | 0.481278 | no |
| 1 | 0.136033 | 0.000000 | 0.229675 | 0.016168 | 0.029373 | 0.026015 | no |
| 2 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | no |

Fixed aligned 32-image pilot; no per-image threshold adaptation.
Event diagnostics are computed from binary component spike histories after settle, before scoring.
