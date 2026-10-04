# SW_0036: membrane threshold calibration diagnostic

Status: 64-image pilot and selected full320 comparison complete.

Frozen SW_0003 weights are evaluated at inference thresholds .06, .25, .5,
1, 2, 3 and 4. This does not claim a trained improvement: the checkpoint was
trained at .06. It locates a range where actual binary events vary in time and
tests whether restoring event dynamics changes distance-controlled object AUC
or fixed membrane/spike spatial k10 readouts.

For every threshold report binary event rate, constant-history fraction,
temporal standard deviation, membrane/gated-spike/binary event macro AUC at
fixed pair distance, and all three mask metrics. Same frozen checkpoint,
validation IDs 1320–1383, 256 steps, settle 64, sigma1.5 and k10. GT is used
only for diagnostic AUC and evaluation.

This is one input to the next training choice. It is not a strict gate and is
run alongside the adaptive-count classifier baseline.

## Pilot result

At vth 2.0, binary event rate falls from 1.0 to .2993, constant-history
fraction from 1.0 to .00066, and temporal std rises from 0 to .4472. Frozen
gated-spike spatial k10 improves on all three pilot metrics from
`.487071/.174560/.160534` to `.494124/.182644/.170329`. Binary distance AUC
is only .5225 and membrane AUC falls, so restoring events does not by itself
create strong object signal. Thresholds 3–4 become too sparse and degrade the
spike readout. Confirm only .06 and 2.0 on full validation; any training test
must initialize/train with the chosen threshold rather than claim this frozen
inference intervention as a learned improvement.

## Full320 confirmation

Vth 2.0 preserves the pilot direction on gated-spike spatial k10:

| Frozen threshold | Event rate | Constant fraction | FG-ARI | FG IoU | Object IoU |
|---|---:|---:|---:|---:|---:|
| .06 control | 1.0000 | 1.0000 | .479762 | .185651 | .164179 |
| 2.0 | .2988 | .00070 | .487351 | .192635 | .172932 |

All three spike-mask metrics improve, while membrane readout changes from
`.494639/.191070/.170568` to `.471315/.193961/.167694`. Distance-controlled
binary/gated-spike AUC at vth2 is only `.5531/.5553`, below the saturated
control's gate-driven `.5863`. Thus vth2 repairs actual event dynamics and
improves the spike readout modestly, but does not solve weak object identity.
Proceed to matched training at vth2 as SW_0037, preserving both spike and
membrane controls.
