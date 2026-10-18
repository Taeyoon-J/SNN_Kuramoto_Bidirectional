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

Actual GPU3 worker2337998/start448637534 began after210 seed2 completed all
four prediction shards. NVIDIA ownership and actual argv/start identity are
recorded in started_v1/actual_GPU_start.json. This starts frozen-core feature
extraction; it does not yet establish that classifier Adam updates have begun.
The earlier GPU2 attempt yielded before extraction to respect another owner.
SW0212 is independently queued to compare learned foreground after completion.

Completed, gate failed: primary128 .034678/.445474/.044414 versus original
.788414/.445474/.674335. All DEV images use a single-center fallback. Actual128
Adam updates and96 endpoints independently audited. TRAIN-only diagnosis also
finds no effective center detections, despite offset error1.5185 pixels versus
zero-offset4.2434 and mean-input4.7278. These are TRAIN diagnostics, not gains.
See RESULT_KO.md and results_archive/head_diagnosis_v1/summary.json. Incumbent
full320 three-seed mean remains .848615/.592533/.672047.
