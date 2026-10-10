# SW0134 native-spike competitive binding

Prospective model preparation; no server training or evaluation has started.
The unchanged SW0133 pilot must finish before this experiment is launched.

This candidate replaces QCC-seeded assignment with an explicitly adopted
Slot Attention-style binder over complete, raw actual spike histories. The
native dynamics remain intact. Three matched arms will distinguish downstream
signal use, joint representation learning, and learned-readout effects. Both
the new spike-derived masks and original QCC readout will be reported.

See [registered protocol](protocol.json) and
[design rationale](../SW_0133_soft_partition_rgb/design_review_after_soft_pilot_20261009.md).
Synthetic helper tests establish implementation properties only. They do not
prove gating usefulness, scaling improvement, or superiority to Slot Attention.
