# SW0108: fixed-checkpoint gate contribution screening

Use SW0097 seed0 positive_frozen and the original native validation gamma
cache on IDs1320-1639. Compare the baseline with five separate rollout
interventions: zero Kuramoto coupling, constant sinusoidal gate, no
dendritic retention, no membrane retention, and forced-on actual events.
The final classifier still consumes actual component spikes. These are
checkpoint-sensitivity tests, not matched retraining ablations.

## Invalid first attempt and correction

The first evaluator command omitted `--membrane-vth .06`, inheriting the
shared evaluator default2.0. Baseline and intervention scores from that
attempt cannot be used under the fixed contract. The queue was stopped;
all recorded PIDs were confirmed absent before the entire output tree,
including logs and lock, was preserved on Frontier at
`trained_models/SW0108_gate_contribution_screen_attempt1_vth2`.
The old preflight and halted queue state are archived within that tree.

Corrected local code supplies .06 explicitly and requires all320 baseline
per-image metrics and means to match the registered SW0097 seed0 record
within1e-10 before any intervention may start. The baseline record SHA is
`009da533e0f3e7fa86d9840a85d87640d9424acf660512f64232ad1669fe61e0`.
The existing server record in
`trained_models/SW0097_graph_adaptation/seed0_positive_frozen/evaluation.json`
was independently confirmed to have that exact SHA; no reference-file
transfer is necessary.

## Deployment status

2026-10-08: user explicitly authorized corrected source transfer and testing.
Deployment and validation are now permitted; verify live queue state before
reporting that evaluation has started.

Historical blocker: corrected code was prepared locally but execution had not restarted.
Automatic approval review rejected transferring the four corrected source
files to Frontier, citing missing explicit source-transfer authorization
and prohibiting a different transfer route. An explicit user approval
request was subsequently approved. No transfer workaround was used while
the request was pending.
After authorization, deploy the frozen files, run server unit tests and
actual-model preflight, then start the baseline-first exclusive GPU queue.
Foreign GPU jobs must not be shared or stopped.
