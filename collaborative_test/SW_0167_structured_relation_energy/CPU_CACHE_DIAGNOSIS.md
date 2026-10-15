# First eight TRAIN rows: CPU arithmetic diagnosis

Read-only server inspection verified source/context manifests, ordered ledgers, selected RGB bytes and spike/Q/A payload hashes for the first eight registered TRAIN cache rows. Both source-state receipts use the same SW0161 digest family and equal `b205d21a917299089b9b5d373eb94c9eb5eb971471e9428072e0fc0229c70749`. The recorded source coupling gain is 7.623262882232666. No mask/GT arrays, validation data, rollouts or optimizer updates were used.

Production CPU Q was recomputed in the original B4 groups. Its maximum absolute difference from the cached GPU Q ranged from 0.0000053644 to 0.0000063777. All eight rows had **zero** threshold-edge disagreements at 0.50, zero QCC label differences and zero foreground-support differences. These observations do not explain an accuracy loss or establish cross-backend bitwise equivalence.

The new solver exactly matched SW0164 energy, labels and complete merge logs on all eight rows using identical cached Q/A and the same CPU-derived support. However, the pair-feature constructor rejected all eight CPU reconstructions against the GPU cache at its unchanged 0.000002 consistency tolerance. Therefore **full preflight failed**: solver-only parity must not be presented as a completed model preflight.

The next arithmetic check compares GPU B4, GPU B1 and cached Q under an idle-GPU ownership lease before attempting the real zero-residual MLP path. The tolerance, cached Q, original source and scientific selection gate remain unchanged. No SW0167 training has started; SW0163 final scoring still determines the next scientific experiment.

Evidence: `results_archive/train_cache_cpu_diagnosis_20261015/server_records.json`. The reproducible read-only script is `audit_train_cache.py`.
