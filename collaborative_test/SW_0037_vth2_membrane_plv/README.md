# SW_0037: train membrane-PLV core with non-saturated vth 2.0

Status: matched seed-0 training and fixed full320 validation complete; mixed
result, not promoted to seeds 1/2.

## 왜 바꾸는가

SW_0034 found that SW_0003's post-settle binary events are constant one.
SW_0036 changed only frozen-checkpoint inference threshold: vth 2.0 restored
event rate .299 with almost no constant histories and improved all three
gated-spike spatial readout metrics on both the 64-image pilot and full320.
That intervention did not train the model for the new dynamics, so this test
trains the matched SW_0003 membrane-PLV configuration from initialization with
vth 2.0.

## Controlled comparison

- Control: SW_0003 seed 0, 40 epochs, membrane PLV, vth .06.
- Candidate: same data, seed, initialization route, epochs, optimizer, losses,
  graph, gating, membrane range and classifier controls; only vth is 2.0.
- First evaluate actual event rate/constant fraction and stage signal on
  validation. Then compare fixed membrane/spike spatial k10 and the frozen
  adaptive-count classifier baseline.
- GT is evaluation/diagnosis only. No reference test or seed 1/2 until seed-0
  validation justifies continuation.

This experiment targets a proven spike-dynamics defect. It may still fail if
the upstream object signal remains too weak; threshold calibration alone did
not materially increase distance-controlled AUC.

`evaluate.sh` runs the fixed full320 stage/event diagnostic first, then the
same adaptive-count reference suite with `--membrane-vth 2.0`. It starts only
after `core.pt` exists and never selects from the reference-test split.

## Full320 result

Training at vth2 repairs the original event saturation: post-settle binary
event rate is `.546981`, constant-history fraction is `.003229`, and temporal
standard deviation is `.471698`. However, distance-controlled object AUC is
only `.511276` for membrane, `.514285` for gated spikes, and `.517934` for
binary events. This is weaker than both the original vth.06-trained continuous
signals and the frozen vth2 intervention, so valid events alone did not create
object-specific grouping.

| fixed spatial k10 readout | FG-ARI | foreground IoU | object IoU |
|---|---:|---:|---:|
| SW_0003 membrane, trained vth.06 | .494639 | .191070 | .170568 |
| SW_0037 membrane, trained vth2 | .478921 | .200110 | .174031 |
| SW_0003 gated spike, inference vth2 | .487351 | .192635 | .172932 |
| SW_0037 gated spike, trained vth2 | .461878 | .204486 | .174888 |

Both trained-vth2 readouts trade lower FG-ARI for higher IoUs, so this is not
an all-metric improvement. The selected adaptive count rows also do not rescue
the tradeoff: gated-spike eigengap scores `.461532/.199782/.142632`, while the
best count MAE is membrane-spatial thresholding at `1.763` with
`.473776/.195471/.121242`. No reference-test evaluation or seed expansion.
The next test keeps vth2 dynamics but directly puts component-spike synchrony
on the gradient path in SW_0038.
