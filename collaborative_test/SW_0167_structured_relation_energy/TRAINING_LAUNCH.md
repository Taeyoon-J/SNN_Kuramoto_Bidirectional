# Fixed two-objective training queue, 2026-10-15

After the completed real-eight GPU admission, the actual full 512-image CPU preparation passed with CUDA uninitialized. Both full caches, ordered TRAIN IDs, frozen source checkpoint, original-B4 CPU source foreground, feasible targets and first-eight receipt parity were verified. No VAL images or masks were used. Five focused trainer/queue CPU tests also passed on the server; Sol5.6 independently reviewed the final trainer and queue.

Two independent owner-guarded queues are registered, using only exclusively idle GPUs:

| Objective | Supervisor PID | Worker PID | Leased physical GPU |
|---|---:|---:|---:|
| Structured partition | 3648428 | 3648536 | 2 |
| Matched local margin | 3648429 | 3648525 | 1 |

At the archived launch observation the worker processes were alive and preparing their inputs before CUDA allocation; **an optimizer update was not yet verified**. Queue status alone is insufficient to claim GPU training progress. Later durable boundaries and actual GPU ownership provide that evidence. Foreign GPU0 PID2653200 was left running.

Both arms use the same 512 TRAIN images, fixed eight-pass order, identical fresh head initialization, 256 logical B16 Adam updates with micro4, learning rate3e-4 and gradient clip1. Only the loss differs. Every completed update has an atomic resumable boundary. Final validation checks exact final head/boundary agreement and all four Adam parameter step counters at256. Neither encoder nor physical source gates are optimized in this head-only supervised pilot.

Both final endpoints must freeze before shared-rollout fixed320 evaluation. Compare the structured candidate against both SW164 seed1 and the matched local control, using the registered paired confidence intervals and exact foreground preservation gate. This launch is not evidence of an improvement, data scaling or gate causality; SW165 remains incumbent.

Trainer SHA256: `3c69d9acb505bf635f72309505c1bf9fe3d951791ddeafaa22e813cda6789642`. Queue SHA256: `0359d2ff800157dd2313f604b9fdf908cd31a89e5559199669b12a049f14126a`.

Evidence: `results_archive/training_launch_20261015/server_records.json`. Server runtime: `results_archive/training_queue_20261015/` and `results_archive/training_20261015/`.
