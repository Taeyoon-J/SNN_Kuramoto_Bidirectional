# SW0211 spike-anchor center assignment

Registered after17 local and17 Linux tests, AST/CLI checks and actual source/
TRAIN32/native64 head-and-decoder CPU admission. Supervisor2269832 is waiting
for an exclusive GPU0..3 lease. No improvement is claimed at registration.
See DESIGN_KO.md. Core stays frozen and the spike foreground is preserved;
only the new center-voting readout learns from TRAIN32 masks.

Tests cover real production correlation descriptor layout/time invariance,
disconnected-instance voting, exact foreground preservation, GT-free decoder,
actual Adam resume, changed-data rejection and GT-last scoring. A separate
synthetic TRAIN-only budget smoke passed; its results are not research-model
performance. Actual cache extraction and head training begin only after a lease.

The actual source/model seed is2, verified from the immutable admission and
checkpoint SHA. The generic dispatcher receipt's seed0 label is not passed to
the worker and is explained in results_archive/registration_provenance_v1.json;
original records are preserved and no job was restarted to correct that label.

Read-only waiter2272841 tracks the exact supervisor PID/start identity,
observes actual GPU start and independently audits completed head/Adam/RNG
boundaries, TRAIN normalization, frozen source and96 mask endpoints. This is
a supervised pilot, not a fresh70k or full320/three-seed .90 result.
