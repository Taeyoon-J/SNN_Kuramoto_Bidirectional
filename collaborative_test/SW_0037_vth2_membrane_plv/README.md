# SW_0037: train membrane-PLV core with non-saturated vth 2.0

Status: planned matched seed-0 training.

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
