# SW0169 actual CPU preparation

2026-10-15. The normalized slow-membrane/binary-hard-reset opt-in implementation and source calibration leaf files were installed immutably on Frontier. The registered native source files and SW0168 were not changed.

Root independently executed all 11 leaf/rollout contract tests locally and on the server: PASS, no skips. Coverage includes binary reset versus emitted amplitude, zero-gate V/E-history hold with actual S=0, active-gate CPU-float64 RMS, fixed tau/b initialization, B2 source six-field/production-Q parity, candidate/control bitwise forward parity, component folding, actual 1024-step boundary960 and tail credit, and actual surrogate carry gradients versus detached control.

Actual `prepare_calibration_cpu()` passed on the server with CUDA uninitialized, no GT and no VAL read. Its source/cache/RGB bindings resolve exactly the first eight ordered TRAIN IDs: 68304,49032,70041,7829,16499,31675,67612,54671, in two contiguous B4 groups. Trace probe positions in the earlier SW0168 module are not image-row indices. Source cache and context digest families stay separate.

This preparation is not the full-size GPU source-calibration parity, mixed-event/gradient admission, training, evaluation, or a positive performance result. Those stages remain required before the pilot may expand. SW165 remains the incumbent.

Evidence: `results_archive/leaf_CPU_20261015/server_records.json` (full server test output, source proof, installed file hashes). Physical generated output is under `collaborative_testing/generated/`; canonical results access is a junction. No dataset/checkpoint/key was transferred or published.
