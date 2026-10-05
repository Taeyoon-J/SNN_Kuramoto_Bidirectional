# SW0057: fixed multi-readout evaluation for SW0055

This standalone, post-training evaluation uses each SW0055 seed's `core.pt` and the fixed HDF5-aligned validation scenes, IDs 1320-1639, with the aligned gamma tensor and manifest from SW0042. The saved manifest contract is `image_ids=[1320,1639]`, `shape=[320,8,256]`, `dtype=torch.float32`, and `patch_grid_size=16`; the evaluator also requires the manifest gamma SHA256 to match the loaded gamma file. Full evaluation uses all 320 rows. The four-image preflight is allowed only as the first four rows of that exact aligned tensor. Inference is fixed to the long window, T=1024 and settle=512, with the SW0042 model configuration (shared dendritic projection, graph spatial decay .35, geodesic steps 3, factorized Kuramoto dynamics, membrane threshold .06).

Two readouts are registered before scoring:

1. Spike synchrony connected components at threshold .50, minimum component size 2, and largest component treated as background. This is the SW0055 primary readout.
2. Absolute membrane correlation multiplied by the 16x16 Gaussian spatial kernel (sigma 1.5), followed by the existing spectral k=10 readout from SW0027/SW0028.

SW0051 defines several distinct frozen-foreground spectral variants (`freeze`, `restrict_k10`, and `restrict_dynamic`) and reports tradeoffs; it does not designate one unique variant as the fixed readout. SW0057 excludes that family instead of selecting among its variants after seeing results. Both included predictions are formed without target masks. The evaluator asserts that both prediction sets are present before it opens the HDF5 mask dataset; masks are then used only for scoring.

Run a four-scene real-checkpoint preflight for each seed before its full evaluation. These scripts do not modify training or the SW0055 scheduler. Only GPUs 0 and 1 are accepted; preflight and full evaluation refuse to overlap a compute process. The preflight marker binds checkpoint, gamma, manifest, evaluator, and imported readout/model code hashes. Changed assets or code require a fresh marker, and a stale marker is never overwritten.

```bash
collaborative_test/SW_0057_fixed_multireadout/preflight.sh 0 0
collaborative_test/SW_0057_fixed_multireadout/evaluate.sh 0 0
collaborative_test/SW_0057_fixed_multireadout/preflight.sh 1 1
collaborative_test/SW_0057_fixed_multireadout/evaluate.sh 1 1
collaborative_test/SW_0057_fixed_multireadout/preflight.sh 0 2
collaborative_test/SW_0057_fixed_multireadout/evaluate.sh 0 2
```

Each seed report and log is written under its existing SW0055 model directory with overwrite refusal. Once all reports are present and validated, create a fixed three-seed mean/sample-SD summary:

```bash
/Data0/kevinswk/envs/snn/bin/python collaborative_test/SW_0057_fixed_multireadout/summarize.py \
  --model-root /Data0/kevinswk/patch_v2_sw/trained_models \
  --output collaborative_test/SW_0057_fixed_multireadout/results/sw0057_long_3seed.json
```

The summary requires all three fixed seed reports and checks their split, window, readout, and provenance contract. It writes a companion Markdown table and does not tune thresholds or choose a readout based on validation outcomes. This preparation task did not launch the evaluation.
