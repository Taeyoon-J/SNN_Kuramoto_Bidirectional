# SW_0020: graph teacher on actual spikes

Status: matched seed-0 10-epoch training and fixed 320-image validation complete.
The spike-source loss was differentiable but did not improve the priority FG-ARI.

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

After training, `evaluate.sh` uses the same SW_0006 component-product spike
classifier and the SW_0011 signal-flow diagnostic on the fixed validation
IDs. It refuses to run until epoch 10 and the checkpoint are present.

## Result and interpretation

The training log reached epoch 10/10 and saved the checkpoint. The final
total training loss was 2.804146; the raw graph-teacher term was 1.160337.
Validation used IDs 1320-1639, actual spike-derived masks, and the identical
threshold sweep used for SW_0004. At the same component-product threshold .50:

| Metric | SW_0004 membrane-source teacher | SW_0020 spike-source teacher |
|---|---:|---:|
| Patch FG-ARI | .195269 | .122560 |
| Patch foreground IoU | .275711 | .260356 |
| Patch matched-object IoU | .238803 | .287220 |
| Predicted groups/image | about 31 | 144.21 |

The object-IoU gain is retained as a potentially useful signal, not dismissed
because other metrics fell. However, the priority FG-ARI and foreground IoU
worsened and object fragmentation increased sharply. In the alternative
aggregate readout, the highest FG-ARI was .185819 at threshold .80, with
foreground IoU .270110 and object IoU .221173. Thus no tested readout beats
SW_0004 on priority FG-ARI, so this exact weight-.1 spike-source training is
not the leading follow-up. Full sweep: `validation_results.json`.

On the first 16 validation images, the phase-to-spike affinity correlation
was .805587 (SW_0004: .816019). Spike synchrony same/different-object
means were .359067/.180711 (SW_0004: .382151/.212295). These diagnostics
do not show a clear improvement in theta-to-spike transfer. Mean actual
spike rate fell from .419544 to .195464. This association suggests that
weight .1 may be too strong, but no causal claim follows from one seed.
See `signal_flow.json`.
