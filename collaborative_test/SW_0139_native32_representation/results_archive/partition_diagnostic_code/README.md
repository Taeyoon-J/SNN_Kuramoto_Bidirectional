# SW0139 seed1 partition-error diagnostic

This is a byte-identical archive of the read-only diagnostic and its local tests. It decomposes partition errors for the completed seed1 SW0139 control and context-residual predictions against source97. It does not change prediction or scoring contracts and does not establish a causal-layer claim.

On the server, run with the repository path explicitly set (especially when feeding the script on stdin):

```sh
PATCH_V2_REPO=/Data0/kevinswk/patch_v2_sw python - --output /Data0/kevinswk/patch_v2_sw/collaborative_test/SW_0139_native32_representation/results_archive/partition_diagnostic_seed1.json < /path/to/diagnose_repr32_partition.py
```

The diagnostic validates the SW0139 partial-complete report, the control/context prediction protocols and hashes, SW0139 training/checkpoint provenance under `trained_models/SW0139_native32_representation`, and the source97 fixed320 prediction before it opens the HDF5 mask slice for IDs 1320 through 1639. It then checks the saved production metrics against a native32 replay and writes one create-only JSON. It forces `CUDA_VISIBLE_DEVICES` empty and sets OMP/MKL thread counts to one.

The collector `collect_repr32_partition.py` in the private deployment folder reads that one JSON, checks its fixed seed1/320/32x32/GT-scope schema, and emits a SHA256-bound base64 transfer payload.
