# SW0058 stage-1 stopping result

Fixed validation IDs 1320-1639; T1024/settle512; spike CC threshold .50.
The candidate and baseline use the same 1,000-scene gamma, LR, epochs, and optimizer recipe.

| Metric | Candidate s0 | Baseline s0 | Delta s0 | Candidate s1 | Baseline s1 | Delta s1 |
|---|---:|---:|---:|---:|---:|---:|
| fg_ari | 0.614743 | 0.701538 | -0.086795 | 0.574495 | 0.592166 | -0.017671 |
| foreground_iou | 0.567100 | 0.610958 | -0.043858 | 0.334240 | 0.345094 | -0.010854 |
| matched_object_iou | 0.434182 | 0.489420 | -0.055238 | 0.334951 | 0.421391 | -0.086440 |

Seed2 was not launched because both completed seeds were lower on every primary metric.
This rejects complete removal of the phase-primary term; it does not reject smaller reweighting or a structural gate-transduction change.
