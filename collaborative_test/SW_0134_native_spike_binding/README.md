# SW0134 native-spike competitive binding

The scientific code bundle has been deployed with exact file/dependency hashes.
Independent local and server CPU suites both passed 13 tests, including real
throwaway optimizer updates with source-shaped synthetic assets for all three
arms. Actual source checkpoint provenance was validated for seeds 0/1/2.
Actual-source GPU preflight and main training have not started.
The unchanged SW0133 pilot is complete; its registered expansion gate failed.

See [deployment evidence](scientific_bundle_deployment_review_20261009.json).

This candidate replaces QCC-seeded assignment with an explicitly adopted
Slot Attention-style binder over complete, raw actual spike histories. The
native dynamics remain intact. Three matched arms will distinguish downstream
signal use, joint representation learning, and learned-readout effects. Both
the new spike-derived masks and original QCC readout will be reported.

See [registered protocol](protocol.json) and
[design rationale](../SW_0133_soft_partition_rgb/design_review_after_soft_pilot_20261009.md).
Synthetic helper tests establish implementation properties only. They do not
prove gating usefulness, scaling improvement, or superiority to Slot Attention.
