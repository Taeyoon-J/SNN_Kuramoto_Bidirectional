# SW0238: completed six-case native64 event diagnostic

All three fixed accepted232 seed checkpoints were checked on two actual TRAIN images each. Source, checkpoint and row bindings and reported integer-count arithmetic were validated. No evaluation labels or optimizer updates were used.

Positive-gate event occupancy was 0.996797/0.997802 for seed0, 1.0/1.0 for seed1 and 0.993265/0.993510 for seed2. On these two seed1 cases, the final spike tensor was bitwise equal to the gate. In the other four cases, observable event suppression was at most 0.006735. All observed cases used identical gates across components.

This supports investigating event saturation: for these traces the membrane/event operation mostly passes the gate. It does not prove that membrane removal preserves segmentation, that all inputs are saturated, or that a threshold change improves performance. Raw traces were not retained, so this archive validates scalar consistency, not independent trace recomputation. A next causal test should compare the actual mask outputs under an explicit event bypass on the same frozen model and input; any subsequent learned intervention needs held-out metric confirmation.

The ongoing70k training is unchanged. This diagnostic produces no new segmentation score and does not replace the audited SW0227 full320 incumbent. Private dataset and sample metadata are omitted from this public aggregate report.
