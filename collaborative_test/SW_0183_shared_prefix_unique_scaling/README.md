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
Three prediction supervisors and one GT-last scoring supervisor are registered.
They wait for training, freeze both arms for all three seeds, then score fixed320.
All 29 server tests and the three executable module CLI checks passed.
[Evaluation CPU receipt](results_archive/cpu_v2/evaluation_cpu_receipt.json),
[evaluation launch receipt](results_archive/start_v1/evaluation_launch.json).
The score module is a library called by score_queue, not a standalone CLI.
No fixed320 result is available. The SW0181 incumbent is intact. This is
supervised continuation, not a fresh native32/70k or unsupervised result.

All seeds passed the first two actual GPU proposals' emitted-S gradient,
four-family gradient/proposed-delta, same-RNG full-forward parity, and actual
Adam-step checks, and continued beyond proposal 2:
[GPU admission receipt](results_archive/start_v1/first_two_GPU_admission.json).
Positive proposed gradients/deltas establish a training path; they do not prove
the accuracy contribution of each gate or imply that a rejected proposal was
committed.
