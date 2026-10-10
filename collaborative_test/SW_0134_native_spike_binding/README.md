# SW0134 native-spike competitive binding

Latest: all three pilot training children were stopped when foreign PID2517185
entered their reserved GPUs. No checkpoint was saved; partial-update count is
unknown. Original state, empty logs and empty output folders were preserved.
An unchanged-code recovery queue (supervisor2566458) is verified alive and
waiting for exclusive GPUs; it restarts from the same registered warm32
state, not an interrupted checkpoint. No new score or scientific rejection.
See [recovery evidence](owner_interruption_recovery_20261009.json).

Pilot queue status: `running`. 3 child processes were verified live in /proc at the recorded observation.
The initial launch failed before its first update because direct script execution
could not import the repository package. Failure logs/state were preserved;
all child stages now use Python module execution. Seven server CPU checks,
including the actual child entrypoints with PYTHONPATH removed, passed before recovery.
No performance result or model promotion is available. See [pilot evidence](pilot_launch_review_20261009.json).

The scientific code bundle has been deployed with exact file/dependency hashes.
Independent local and server CPU suites both passed 13 tests, including real
throwaway optimizer updates with source-shaped synthetic assets for all three
arms. Actual source checkpoint provenance was validated for seeds 0/1/2.
All three actual-source GPU preflights passed canonical validation, including
native full1024 parity, warm32 provenance and all three real disposable arms.
Seed1 passed after a technical recovery preserving the original GPU-owner
interruption evidence and scientific recipe. Its supervisor/child are terminal.
The earlier preflight-only observation preceded pilot launch. See [passed preflight evidence](results_archive/passed_preflights_20261009/summary.json)
and [exact first-pass evidence](results_archive/first_pass_preflights_20261009/summary.json).
The recovery preserved the original state/log hashes and scientific recipe;
see [recovery process evidence](seed1_recovery_launch_20261009.json).
The unchanged SW0133 pilot is complete; its registered expansion gate failed.

See [deployment evidence](scientific_bundle_deployment_review_20261009.json).

The separate preflight queue was subsequently deployed and launched after the
expanded local and server suites both passed 15 tests. Supervisor PID 2427541
was confirmed live in `/proc`. At launch observation all four GPUs had a foreign
compute owner, so seed0 was waiting for an exclusive GPU and seeds1/2 were queued.
The queue reserves an available GPU automatically; seeds1/2 depend on a valid
seed0 preflight. This is a queued GPU diagnostic, not completed main training.
See [launch and process evidence](preflight_queue_launch_20261009.json).

The queue subsequently reserved GPU0 and started seed0 child PID2483147.
Both that child and supervisor PID2427541 were confirmed live in `/proc`.
See [actual GPU start evidence](gpu_preflight_start_20261009.json). Seeds1/2
remain dependent on a canonically valid seed0 result; no performance result
or model promotion is implied by this technical diagnostic.

This candidate replaces QCC-seeded assignment with an explicitly adopted
Slot Attention-style binder over complete, raw actual spike histories. The
native dynamics remain intact. Three matched arms will distinguish downstream
signal use, joint representation learning, and learned-readout effects. Both
the new spike-derived masks and original QCC readout will be reported.

See [registered protocol](protocol.json) and
[design rationale](../SW_0133_soft_partition_rgb/design_review_after_soft_pilot_20261009.md).
Synthetic helper tests establish implementation properties only. They do not
prove gating usefulness, scaling improvement, or superiority to Slot Attention.
