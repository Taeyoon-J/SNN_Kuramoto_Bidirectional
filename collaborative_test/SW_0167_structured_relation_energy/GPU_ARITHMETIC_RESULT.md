# GPU arithmetic check and feature-mode decision

The owner-guarded GPU1 probe completed on the same first eight registered TRAIN cache rows. Four CPU checks passed before launch; source/order/row hashes were verified under an exclusive lease. No source rollout, mask/validation read or optimizer update occurred.

| Feature consistency comparison | Rows passing unchanged 2e-6 guard |
|---|---:|
| Per-image GPU rho against cached GPU-B4 Q | 0/8 |
| Per-image GPU rho against recomputed GPU-B4 Q | 0/8 |
| Per-image GPU rho against same-mode GPU-B1 Q | 8/8 |

Recomputed GPU-B4 Q matched cached Q exactly on rows4–7; rows0–3 differed by at most 3.04e-6. GPU-B1 Q differed from cached Q by about 5.36e-6–6.20e-6. All comparisons had zero threshold-edge disagreements at0.50. This does **not** justify replacing cached Q: the logit-weighted baseline energy changed by as much as49.1614 when computed from B1 Q. The first cache-generation batch and later batches also use different tensor construction paths; that observation is not a proven causal explanation of the numerical differences, and no further stride exploration is needed for this pilot.

Sol6.1 reviewed the actual implementation and report and approved an explicit separation of arithmetic contracts. Keep cached TRAIN Q/A/gain and baseline E0/wA unchanged. Derive rho from the identical saved component spikes using a declared GPU-B1 mode. A new feature-building API combines these independently provenance-validated inputs without asserting cross-mode bitwise factorization. The existing same-mode wrapper and2e-6 check stay unchanged. Both loss arms use identical feature bytes and modes. TRAIN foreground remains production CPU-B4 QCC from saved spikes; VAL uses actual native-B1 source output and its existing CPU-QCC readout.

Before training: verify new-builder parity with the old wrapper on same-mode inputs; retain source/cache/order hashes; require exact corrected-E0/source-support/merge-log/label parity against the cached-Q baseline on all eight real TRAIN rows at zero residual; verify the two-update gradient credit for both objectives. These checks are pending implementation. Neither this arithmetic probe nor CPU synthetic tests are performance evidence. The scientific choice of SW167 follows SW163's failed registered gate, not favorable numeric variants.

Evidence and ownership receipts: `results_archive/gpu_arithmetic_diagnosis_20261015/`. The completed probe code remains immutable; later admission uses a separate entrypoint.
