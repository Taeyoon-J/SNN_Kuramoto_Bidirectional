# Checkpoint GPU admission correction

The original checkpoint runtime signature includes the physical `CUDA_VISIBLE_DEVICES` string. The original supervisor could select another idle GPU after foreign-owner interruption, leading to a terminal runtime mismatch even when both GPUs have the same model. This correction preserves that strict signature and chooses only the original physical GPU when a valid committed checkpoint exists. Before the first commit, any of the four exclusive idle GPUs remains eligible. Predictions can use any exclusive idle GPU.

`stage_queue_v2.py` is a separate operational supervisor with its own runtime directory. The original eight sealed SW0148 files, scientific recipe, checkpoints, and SW0146 lineage remain unchanged. Corrupt checkpoints, invalid CUDA metadata, and invalid committed update counts fail before GPU reservation. A busy checkpoint-bound GPU causes a wait rather than migration.

All 19 Linux CPU tests passed without skips: the original ten checkpoint/pipeline tests and nine v2 admission tests. The new tests exercise the actual GPU reservation loop with simulated occupancy, pinning, cancellation, affinity forwarding, malformed checkpoint rejection, and completed-result reuse. These checks do not prove production CUDA resume or segmentation improvement; those remain pending actual training.

The first supervisor was frozen and identity-checked before replacement. It had no training children. Its original state and switch evidence are archived under `results_archive/server_start_v2_20261012/`. The replacement supervisor is PID543798. Both training tasks were queued at launch; no new segmentation score is claimed.

Use the v2 entry point for this run:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v2 --dry-run
```

Do not launch another supervisor while the recorded process is live. Observe the existing `results_archive/queue_v2_20261012/state.json` and actual process first.
