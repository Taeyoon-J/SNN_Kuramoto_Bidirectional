# SW0183 — shared-prefix unique-data comparison

Preregistered successor to the SW0182 technical reproduction stop. Each seed
executes one 32-proposal common prefix, then forks the exact model/Adam/RNG state
into 32-proposal repeat512 and new512 branches. This yields two logical
64-proposal arms with 96 proposals of actual work per seed.

See [the registered design](DESIGN_KO.md). The trainer and exclusive queue passed
12 local/server tests, including real populated-Adam transaction independence,
actual SW0148 checkpoint save/load with exact RNG restoration, and arm schedules.
Actual server admission verified all three fresh source initializations and the
registered cache hashes before CUDA, plus a separate module CLI subprocess:
[CPU receipt](results_archive/cpu_v1/training_cpu_receipt.json).

All three seed workers have started on GPUs 1/2/3 and saved their first shared
proposal. [GPU start observation](results_archive/start_v1/gpu_start_observation.json),
[supervisor launch receipt](results_archive/start_v1/training_launch.json).
Prediction/scoring adapters are being implemented while the shared prefix runs.
No fixed320 result is available. The SW0181 incumbent is intact. This is
supervised continuation, not a fresh native32/70k or unsupervised result.
