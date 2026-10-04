# SW_0020: graph teacher on actual spikes

Status: two-image gradient diagnostic completed; matched 10-epoch training
running on frontier GPU 0 (PID 1151716, launched 2026-10-03). Validation pending.

## Motivation

SW_0016 separated neighboring membrane patterns but reduced spike object
separation. All prior graph-teacher training applies its objective to membrane
histories, while the evaluated component-product classifier uses actual spikes.
This pilot asks whether applying the **same existing** graph-teacher objective
to actual aggregate spike histories can close that gap.

## Controlled design

The only model-training change is an optional
`--graph-teacher-signal {membrane,spikes}` argument. Default `membrane`
preserves existing behavior. In `spikes` mode, the graph remains detached and
the identical KL/synchrony formula is applied to the actual hard spike tensor;
backpropagation uses the core's existing surrogate spike gradient. No S2NetCore,
Kuramoto, dendritic, membrane, or graph formula is changed.

First, compare raw loss and gradient norms to dendritic/membrane parameters
for both sources on the same two-image batch from the SW_0004 checkpoint.
Only if spike-source gradients are nonzero and finite should a matched fresh
seed-0 10-epoch run be launched against SW_0004's membrane-source 10-epoch
pilot. Keep all other SW_0004 settings identical, including graph-teacher
weight .1, static 64-step phase-driven core, phase PLV, and batch size 16.
Compare both on the same fixed 320-image validation IDs, with spike-derived
masks only. GT masks are evaluation-only.

The two-image check (IDs 1320-1321, the SW_0004 checkpoint) gave raw loss
1.64249 on spikes versus 1.06054 on membrane. Spike-source gradients were
finite and nonzero: `oscillator_dense.weight` norm .280915, bias .253787,
and `tau_m` .002523. The corresponding membrane-source norms were .006926,
.007533, and .005807. This proves the optional spike objective reaches SNN
weights under the current surrogate gradient, but the roughly 40x larger
dense gradient also raises a stability risk. The matched pilot keeps the
same weight .1 and gradient clipping as SW_0004 to isolate the source change;
training and validation will determine whether that was useful.

Reproduction command: `bash collaborative_test/SW_0020_spike_graph_teacher/run.sh 0`
on frontier, after confirming GPU 0 is available. The script records a PID
and writes the training log/checkpoint beneath
`trained_models/SW_0020_spike_graph_teacher_seed0/`.
The running instance used that exact script and PID 1151716. Confirm it is
terminal and the final checkpoint exists before evaluation; do not restart
solely because an SSH observation times out.

## Exact code changes

- `snn_kuramoto_bidirectional/training/train_s2net_core.py`: optional signal
  selector and CLI flag; default unchanged.
- This experiment's diagnostic, run script, and report files. No production classifier
  change.
