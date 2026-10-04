# SW_0042 - HDF5-aligned BIM6 reproduction

This reproduces the SW0041 BIM6 training recipe while using the HDF5-aligned
training gamma from the source run. Seeds 0/1/2 use that run's `gamma_train.pt`
(IDs 0-999); evaluation uses the separately prepared aligned gamma for HDF5
validation IDs 1320-1639. It does not use the peer gamma sequence or peer
targets.

Training matches SW0041: 40 epochs, seed-specific initialization, spike PLV
weight 5, PLV bimodality weight 6, graph spatial decay .35 and geodesic steps
3, membrane threshold .06, branch 4, shared dendritic projection, and 64
training steps with settle 32. Training and evaluation use the SW0043-verified
factorized Kuramoto backend; evaluation also explicitly loads spatial decay
.35. The aligned gamma provenance is the source
run `/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002`;
validation gamma is `/Data0/kevinswk/patch_v2_sw/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt`.
Its alignment provenance manifest is `/Data0/kevinswk/patch_v2_sw/data/SW_0042_hdf5_aligned/manifest.json`; each evaluation copies the manifest into result provenance and uses global IDs 1320-1639 against gamma rows 0-319.

`evaluate.sh GPU SEED short|long` evaluates the same fixed HDF5 IDs with
common spike synchrony thresholds `.05, .10, .20, .35, .50`. Short mode uses
T256/settle64; long mode uses T1024/settle512. Both compute spike endpoint
metrics and, at the same threshold/window, a phase-PLV connected-component
endpoint diagnostic. Phase metrics are explicitly diagnostic and are not the
formal spike score. The validation gamma tensor has only 320 rows, so the
evaluator maps global IDs 1320-1639 to local gamma rows 0-319.

`launch_after_sw0041.sh --dry-run` checks that all three SW0041 watchers wrote
all three evaluation results and finished `peer_long`, that source/data files
exist, that SW0042 outputs do not collide, and that GPUs 0-2 have no compute
processes. If those conditions are not ready yet, `--wait` polls every 20
seconds; `--wait-timeout-seconds N` exits without launching if the gates remain
closed. Once ready, the script rechecks collisions and GPU occupancy, starts
seeds 0/1/2 on GPUs 0/1/2, and records per-seed training and watcher PIDs/logs.
An OS file lock serializes launcher invocations to prevent duplicate starts.
The watcher verifies the checkpoint and training completion marker, then runs
short and long evaluations sequentially. It refuses occupied GPUs and never
terminates existing processes. This preparation did not launch training.

For a single seed, `launch_seed_when_gpu_free.sh GPU SEED PREREQUISITE` waits
until the marker exists and the selected GPU has no compute process, then
starts that seed and its evaluation watcher without touching existing jobs.

## Seed0 validation results

Seed0 completed 40 epochs with final loss `18.82954237`. On the fixed HDF5
validation IDs 1320-1639, the short T256/settle64 sweep's best fixed threshold
was .35, scoring FG-ARI / foreground IoU / matched-object IoU
`.582002 / .600063 / .447548`. For long T1024/settle512, threshold .35 gives
the best reported FG-ARI and matched-object IoU (`.598174 / .605847 / .461779`),
while threshold .20 gives the best foreground IoU (`.596839 / .615626 / .457246`).
Long-window prediction-derived count exact accuracy / MAE is `.1906 / 1.6344`
at .35 and `.2156 / 1.6906` at .50. These are separate per-threshold,
per-metric seed0 observations.

All three seeds subsequently completed. At the common short threshold .35,
the mean is `.399019 / .433654 / .279525` with sample standard deviation
`.199366 / .258816 / .215082`. At the common long threshold .35, the mean is
`.415879 / .418226 / .296118` with sample standard deviation
`.210397 / .249434 / .227312`. Seed2 collapsed to approximately
`.1873/.1355/.0371` short and `.1857/.1352/.0370` long across the threshold
grid, which dominates the variance.

The final training losses for seeds 0/1/2 are
`18.829542 / 17.359897 / 14.268682`; the lowest-loss seed is the collapsed
one. Short-window phase product-PLV mean/std across images is
`.714993/.143289`, `.648663/.170661`, and `.445441/.006269`, respectively.
This is consistent with an upstream, unusually input-insensitive seed2
representation rather than only a threshold or downstream membrane/spike
failure. A scalar PLV mean cannot establish structural invariance, so SW0049
is retained to compare the full stage signals more precisely.
