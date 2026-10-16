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

The registered hash check and failed artifacts remain intact. The next probe
replays proposal 2 from the same saved boundary on one GPU, twice with each
TRAIN512/TRAIN1024 input container, to distinguish execution variability from
a real input/path difference. It reads no evaluation masks. SW0181 remains
unchanged. This is supervised fine-tuning from a pretrained source, not fresh
native32/70k training.
