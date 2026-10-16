# SW0182 — unique TRAIN1024 at the SW0181 proposal budget

This tests whether replacing SW0181's repeated second TRAIN512 pass with the next
512 canonical TRAIN images improves fixed320 masks. Source initialization, loss,
four optimizer families, learning rates, RNG law, actual-loss acceptance and
64-proposal budget remain fixed. See [the preregistered design](DESIGN_KO.md).

The independent server data reference verified 1,024 unique IDs, canonical RGB
alignment, exact preservation of the old 512 RGB/modal rows, and no validation
or reserve reads. It recorded the expected full-array hashes before the candidate
cache was generated: [data reference proof](results_archive/cpu_v1/reference_data_proof.json).

The actual candidate cache was built on Frontier and matched the independent
RGB/modal reference hashes exactly. Its second load reused identical immutable
files. Five data/schedule tests passed locally and on the server:
[cache admission receipt](results_archive/cpu_v1/data_cpu_receipt.json).

The trainer, exclusive GPU supervisors, GT-free predictions and GT-last scoring
are implemented. All 32 local tests passed, including first-32-proposal control
matching, the actual fixed320 aggregation-to-gate connection, and a final GPU
owner/memory check immediately before prediction launch. Four separate module
`--help` subprocesses also passed:
[local CLI admission](results_archive/cpu_v2/local_evaluation_admission.json).

After SSH recovery, read-only inspection confirmed that the timed-out transfer
had not deployed the trainer or launched a queue. The server then passed actual
three-seed CPU admission, all 32 tests, and four module CLI subprocesses:
[training admission](results_archive/cpu_v3/training_cpu_receipt.json),
[evaluation admission](results_archive/cpu_v3/evaluation_cpu_receipt.json).
Three training parents, three dependent prediction parents and a GT-last score
coordinator were registered. Actual workers started on GPUs 1/2/3.

All seeds passed the first proposal's exact control-state check, then stopped
at proposal 2 because the proposed model/Adam digest differed from the archived
SW0181 control. The start-state check at proposal 2 had already passed. Boundaries
retain exactly one committed proposal (accepted counts 1/0/1); predictions and
fixed320 scores were not produced. All owned training/prediction/score processes
are terminal. This is a technical reproduction failure, not a finding about
unique-data benefit:
[failure](results_archive/technical_failure_v1/proposal2_failure.json),
[terminal observation](results_archive/technical_failure_v1/terminal_observation.json),
[committed-prefix audit](results_archive/technical_failure_v1/committed_prefix_audit.json).

The registered hash check and failed artifacts remain intact. A private probe
has now started replaying proposal 2 from the same saved seed1 boundary on its
original GPU2, twice with each TRAIN512/TRAIN1024 input container, to distinguish
execution variability from a real input/path difference. It passed actual CPU
admission and CLI inspection before launch and asserts that the restored state
equals the archived control's proposal-2 start digest:
[CPU proof](results_archive/diagnostic_v1/probe_cpu_admission.json),
[launch proof](results_archive/diagnostic_v1/probe_launch.json),
[exact deployed source](results_archive/diagnostic_v1/probe_source.py).
The source archive records the original private module layout and is evidence;
its embedded module path points to the private deployment location.

The four replays completed. All current Q/Z hashes, pre/post-clip gradient
tensors and proposed model/Adam tensors match exactly across both input
containers and both repetitions. All four reject proposal 2. The archived
control has the same recorded runtime and Q/Z but its proposed state digest
differs. Its context-weight gradient norm differs by about 1.45e-13 and its
parameter-delta norm by about 3.64e-12; the historical rejection decision agrees.
The full archived proposal-2 tensor snapshot is unavailable, so the exact origin
of the historical mismatch remains unresolved:
[replay result](results_archive/diagnostic_v1/proposal2_replay_seed1_attempt01.json),
[archived-control comparison](results_archive/diagnostic_v1/proposal2_control_comparison.json),
[comparison source](results_archive/diagnostic_v1/control_comparison_source.py).

SW0182 closes as a technical reproduction stop with no fixed320/scaling result.
The probe read no evaluation masks and committed no updates. The successor
[SW0183](../SW_0183_shared_prefix_unique_scaling/DESIGN_KO.md) compares two current
arms forked from one exact common prefix, preserving this failed test and its
registered checks. SW0181 remains unchanged. This is supervised fine-tuning
from a pretrained source, not fresh native32/70k training.
