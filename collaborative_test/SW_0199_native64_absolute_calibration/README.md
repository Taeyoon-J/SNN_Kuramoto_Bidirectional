# SW0199 matched native64 absolute calibration

Two independent ready queues compare unchanged QCE and QCE plus absolute-Q
class/pair-balanced Bernoulli calibration. Both resume the same actual TRAIN8
model/Adam/RNG and attempt two new B4 proposals. Production gates and readout
stay fixed; this is a supervised mini, not a70k or three-seed final result.

The hypothesis and fixed extension criteria are registered in DESIGN_KO.md.
No performance result is claimed before actual frozen predictions are scored.
Only aggregate results are public; imagewise GT-derived values remain private.

2026-10-17:19 local tests passed with one Linux-only lock check skipped; all20
Linux tests and2 CLI entrypoints passed. Both arms completed actual CPU admission,
preserving source Adam steps2 and matching model/Adam/RNG start hashes. Separate
all-GPU exclusive-lease queues registered: control1973294, calibrated1973323.
At registration both supervisors were verified live and waiting for an idle GPU.
Receipts are in results_archive/source_tests_v1 and registered_v1.
