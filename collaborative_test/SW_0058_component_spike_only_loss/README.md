# SW0058 - component-spike-only loss ablation

SW0058 tests whether the current mixed objective needs its phase-PLV base term
when the product-combined component-spike synchrony term already uses the same
PLV-family criterion. The sole intended objective change from the SW0053
low-learning-rate, epoch-25 recipe is `--primary-loss-weight 0`; the
component-spike auxiliary remains `--spike-plv-weight 5.0`. Seeds 0/1/2 use
the same 1,000-scene cached training gamma, model settings, optimizer, data
order policy, and 25 epochs as the SW0053 low-LR runs.

This is a loss/model intervention. SW0055 is a separate data-scale control
that changes the number of unique training scenes; SW0058 keeps the SW0053
1,000-scene training data fixed and changes only the primary-loss contribution.

Before training, the gate requires the completed SW0054 seed0 32-image pilot
and recomputes its summary with the current validator. It also requires a
real-gamma one-batch preflight bound to the current trainer/model code, gamma
SHA, and SW0054 raw/summary/validator hashes. The preflight invokes the same
canonical training module as the full run, checks that the saved SC exactly
matches the full 1,000-scene gamma tensor, and verifies a finite loss, zero
weighted primary total, spike-only weighted total, and finite checkpoint.
Preflight and training accept GPU0 or GPU1 only. All output
directories refuse overwrite; partial artifacts are retained for manual
review. No training was launched while preparing this experiment.

The fixed training recipe is LR `3e-4`, 25 epochs, batch 16, 8 feature maps,
256 regions, 4 oscillators per region, 64 recurrent steps, PLV settle 32,
learned graph top-k 32, spatial decay `.35`, factorized Kuramoto backend,
geodesic steps 3/radius 1.5/contrast 2/temperature .5/cap 16, shared
dendritic projection, membrane threshold `.06`, and BIM6 criterion weights.
The exact command-line vector, source gamma hash, dependency code hash, seed,
train IDs 0-999, validation IDs 1320-1639, and SW0053 comparison paths are
written to each seed's manifest before training.

The SW0054 pilot is only a 32-image seed0 diagnostic. It showed a strong loss
under gate permutation while event rate and membrane variance stayed near
unchanged, a large K=0 readout drop, and little effect from carrier
permutation. This motivates testing gate and coupling roles, but is too small
to establish a general mechanism. Its validator result is a launch prerequisite,
not evidence that SW0058 will improve.

The registered comparison is the SW0053 mixed-loss low-LR epoch-25 recipe,
including the seed2 low-LR epoch-25 checkpoint evaluated in SW0052. The primary
comparison is long T1024/settle512 at the predeclared spike synchrony threshold
`.50`, with FG-ARI, foreground IoU, and matched-object IoU reported per seed and
as mean/sample SD. Thresholds will not be selected after seeing results.
Short-window evaluation is diagnostic. The SW0057 fixed long-window membrane
spatial spectral and spike-CC readouts are planned as separately registered
secondary diagnostics; they must not replace the primary endpoint.

Once the SW0054 pilot is validated, an operator may run the real-asset preflight
and launch the three-seed workflow. The launcher runs an actual one-batch
canonical-trainer preflight, trains seeds 0/1 in parallel on distinct GPUs 0/1, then
seed2 on GPU0, and for each seed runs count-4 checkpoint/gamma smoke tests for
both the spike and SW0057 multireadout evaluators before full short/long
evaluation. It validates each phase before writing completion markers and then
creates an exclusive three-seed JSON/Markdown comparison:

```bash
bash collaborative_test/SW_0058_component_spike_only_loss/launch_parallel.sh --dry-run 0 1
bash collaborative_test/SW_0058_component_spike_only_loss/launch_parallel.sh 0 1
```

The dry-run checks the validated pilot, seed output paths, summary path, and
whether any existing preflight is current without training or evaluating. Each
actual training/evaluation phase rechecks its assigned GPU. The launch refuses
existing seed outputs, lock/state, preflight artifacts, and summary files;
failed/partial outputs are left for manual review. `run.sh`, `evaluate.sh`, and
`evaluate_multireadout.sh` are available for explicit phase-level operation.

The canonical one-batch preflight passed with loss `20.36037254`, weighted
primary total `0`, and weighted component-spike total `20.360373`. Seeds 0 and
1 then completed. At the registered long-window threshold, both were lower
than their SW0053 baselines on all three primary metrics. Seed2 was therefore
not launched. The exact stage-1 stopping result is recorded in
`results/stage1_seed0_seed1_stop.{json,md}`. This rejects complete removal of
the phase-primary term under this contract; it does not decide whether a
smaller reweighting or a structural gate-transduction change can improve it.
