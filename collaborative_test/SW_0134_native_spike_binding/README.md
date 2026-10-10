# SW0134 native-spike competitive binding

The scientific code bundle has been deployed with exact file/dependency hashes.
Independent local and server CPU suites both passed 13 tests, including real
throwaway optimizer updates with source-shaped synthetic assets for all three
arms. Actual source checkpoint provenance was validated for seeds 0/1/2.
Actual-source GPU preflight and main training have not started.
The unchanged SW0133 pilot is complete; its registered expansion gate failed.

See [deployment evidence](scientific_bundle_deployment_review_20261009.json).

The separate preflight queue was subsequently deployed and launched after the
expanded local and server suites both passed 15 tests. Supervisor PID 2427541
was confirmed live in `/proc`. At launch observation all four GPUs had a foreign
compute owner, so seed0 was waiting for an exclusive GPU and seeds1/2 were queued.
The queue reserves an available GPU automatically; seeds1/2 depend on a valid
seed0 preflight. This is a queued GPU diagnostic, not completed main training.
See [launch and process evidence](preflight_queue_launch_20261009.json).

This candidate replaces QCC-seeded assignment with an explicitly adopted
Slot Attention-style binder over complete, raw actual spike histories. The
native dynamics remain intact. Three matched arms will distinguish downstream
signal use, joint representation learning, and learned-readout effects. Both
the new spike-derived masks and original QCC readout will be reported.

See [registered protocol](protocol.json) and
[design rationale](../SW_0133_soft_partition_rgb/design_review_after_soft_pilot_20261009.md).
Synthetic helper tests establish implementation properties only. They do not
prove gating usefulness, scaling improvement, or superiority to Slot Attention.
