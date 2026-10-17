# SW0185 — QCE plus spatial spike connectivity

Actual GPU training has started on GPU2; all three seed supervisors and the
three prediction supervisors plus GT-last scoring coordinator are registered.
They use exclusive idle devices and yield to foreign owners.

The objective adds class-balanced same-class component-rho and cross-class
production-Q spatial margins to unchanged QCE. Models, optimizer rates, TRAIN512
two-pass/64-proposal budget, actual-descent accept/rollback and native32 fixed320
S→Q→QCC→SW165 readout remain as registered in [the design](DESIGN_KO.md).
This is supervised continuation from Source097, not fresh70k training.

Server validation passed 20 training tests, then 39 combined training/evaluation
tests and four actual module CLI checks. Seed0's first two actual GPU proposals
passed loss-to-S/four-family proposed gradient/delta, full-forward parity and
Adam checks. The second was rejected and rolled back; proposed credit is not
the same as a committed update, and does not establish gate usefulness.

No new fixed320 score is available. SW181 remains the supervised FG incumbent.
CPU, launch and first-GPU evidence is under `results_archive/cpu_v1`, `cpu_v2`,
and `start_v1`. Source fingerprints are frozen while training is live.

An additional CPU-only completion auditor is registered after scoring. Eight
additional server tests and its actual CLI passed, including arithmetic/gate
verification on pinned actual SW183 vectors. It independently recomputes modal
GT, all three metrics for 1,920 candidate endpoint/image rows and 960 Slot rows,
seed means, paired confidence intervals and registered gates. It waits on the
verified live score producer and never updates the best-model tracker itself.
[Audit launch evidence](results_archive/audit_start_v1/).
