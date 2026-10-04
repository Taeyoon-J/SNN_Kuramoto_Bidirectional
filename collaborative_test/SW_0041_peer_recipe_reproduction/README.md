# SW_0041: Peer BIM6 recipe reproduction

Status: complete for seeds 0, 1, and 2 on GPUs 0, 1, and 2,
respectively. Do not duplicate or relaunch training. This recipe uses the
shared dendritic projection; it does not use per-region projection.

This reproduces the peer BIM6 seed configuration through our training CLI. The
provided training gamma file is the 1000-row training set; the launcher uses it
unchanged and fixes seed, 40 epochs, bimodality weight 6, spike PLV weight 5,
geodesic steps 3, spatial decay .35, membrane threshold .06, 64 training steps,
and branch count 4 to the documented peer recipe.

After each seed's training process exits, run
`watch_and_evaluate.sh GPU_ID SEED TRAIN_PID`. It waits for the process to end,
checks the completion marker and checkpoint, then runs three evaluations in
sequence. Each writes a separate JSON and log under that seed's model folder.
`evaluate.sh GPU_ID SEED MODE` can run one mode directly.

Both modes explicitly use our checkpoint architecture/configuration:
geodesic steps 3, radius 1.5, contrast 2, temperature 0.5, cap 16, membrane
threshold 0.06, shared projection, 256 rollout steps, and settle 64. The
threshold sweep is 0.05-0.50 on our validation and 0.20/0.35/0.50 on peer
validation. `peer_long` uses peer validation IDs 6000-6999 with 1024 rollout
steps, settle 512, and thresholds 0.05/0.10/0.15/0.20/0.35.

- `our_validation`: HDF5 IDs 1320-1639 (320 scenes), scoring the same
  predictions against HDF5 and optionally peer targets through verified
  identity mapping. Peer-target results are cross-target diagnostics only.
- `peer_validation`: IDs 6000-6999 (1000 scenes), also scores both HDF5 and
  peer targets. The peer manifest identifies these as peer validation rows.
- `peer_long`: the same 1000 peer validation rows under long-rollout settings.

The latest unpushed peer long-rollout BIM6 result reports seed FG-ARI
0.6362/0.7100/0.7199 (mean 0.6887). The per-region variant was negative and is
held; this SW0041 reproduction remains shared projection. These peer numbers
are context for the requested long evaluation, not results from SW0041.

This experiment checks code-path reproducibility and keeps evaluation contracts
separate. It does not justify pooling raw peer scores with our HDF5 scores.

## Completed seed results

Formal peer validation (IDs 6000-6999) shows short T256/settle64 means at
threshold .20 of FG-ARI / foreground IoU / matched-object IoU
0.560392 / 0.610688 / 0.404574; threshold .35 gives
0.556789 / 0.609456 / 0.406034, and .50 gives
0.546366 / 0.602581 / 0.402983. For long T1024/settle512, threshold .10 is
best among common rows: 0.574026 / 0.605911 / 0.415638. Per-seed FG-ARI is
0.3700 / 0.6543 / 0.6978, with seed 0 collapsed versus direct peer
reproduction. Thresholds .05 and .15 give means 0.572251 / 0.608677 / 0.413222
and 0.573070 / 0.604528 / 0.415632.

The same predictions scored on our HDF5 validation have best means of
approximately 0.0369 / 0.2502 / 0.1294. This cross-target gap reinforces the
renderer/target contract mismatch; the targets do not affect prediction
formation. Direct peer-exact retraining reproduced seed 0 epoch-1 loss
23.52399481, matching the original 23.523995 to reported precision. These are
transfer/reproducibility diagnostics, not pooled metrics or a claim of
target-independent performance.
