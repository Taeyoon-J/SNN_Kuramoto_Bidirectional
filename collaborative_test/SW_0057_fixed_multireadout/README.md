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

For one-shot automatic execution after SW0055 completes, use the guarded scheduler:

```bash
bash collaborative_test/SW_0057_fixed_multireadout/wait_for_sw0055_and_evaluate.sh --dry-run
nohup bash collaborative_test/SW_0057_fixed_multireadout/wait_for_sw0055_and_evaluate.sh \
  > /Data0/kevinswk/patch_v2_sw/trained_models/SW0057_scheduler_console.log 2>&1 &
```

It waits for `SW0055_unique2500_autolaunch/LAUNCH_COMPLETED` and verifies all three SW0055 seed checkpoints and base evaluations. It then runs seed 0 on GPU0 and seed 1 on GPU1 concurrently, with preflight followed by the full evaluation on each assigned GPU; only after both pass does it run seed2 on GPU0. Each preflight/evaluation phase checks its GPU immediately before invocation. It retries only an artifact-free start race where the GPU became occupied; any result/log or marker produced by a failed phase halts the scheduler and is retained for review. A single-instance lock protects the scheduler, stale locks are removed only when their recorded process is verifiably dead, and any existing/partial SW0057 outputs prevent restart. On success it validates and writes the three-seed JSON/Markdown summary plus a completion marker. GPUs 2 and 3 are never used.

The evaluation completed. Spike CC reproduced the SW0055 primary mean at
`.626965/.423564/.440859`. Membrane spatial spectral raised FG-ARI to `.738478`
with lower seed spread, but foreground IoU/matched-object IoU were only
`.222631/.368848`. Thus the membrane contains stable object-separation signal,
while foreground/background assignment and object overlap are not preserved in
that readout. The fixed result is in `results/sw0057_long_3seed.{json,md}`.
