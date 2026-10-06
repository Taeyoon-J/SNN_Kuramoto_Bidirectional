# SW0056 - Matched-data 2,500-scene Slot Attention candidate

SW0056 is a separate matched-data baseline candidate. It trains from scratch
on HDF5 IDs 0-999 plus 1640-3139 (2,500 unique images) and validates only on
IDs 1320-1639. It does not reuse SW0048 outputs or change the SW0046 transfer
reference.

The fixed budget is ten appearances of each training image: 25,000 real image
exposures, batch size 16, and 1,563 optimizer updates. A deterministic seeded
permutation of all 2,500 IDs is made for each pass; passes are concatenated
before batching. Thus there are 1,562 full batches plus one final batch of
eight real images. Nothing is padded or dropped. The official `build_model`
function passes its `batch_size` argument to a fixed-batch Keras Input, and the
official decoder reshapes with the static image batch dimension. The runner
therefore builds a batch-16 model and, only for the final update, an auxiliary
batch-8 model with verified identical trainable-variable shapes. It copies the
current primary weights to the auxiliary model, computes loss/gradients on the
eight actual samples, and applies those gradients to the primary variables with
the same Adam optimizer state. The final step is not trained on padding.

Official recipe elements are 128x128 full uncropped RGB input normalized to
[-1,1], 11 slots, 3 iterations, reconstruction MSE, Adam at 4e-4 with epsilon
1e-8, and the official warmup/exponential-decay schedule rescaled to warmup
156 and decay 1,563 updates. The official implementation accepts the fixed
batch dimension in `build_model`; see Google's
[object-discovery training loop](https://raw.githubusercontent.com/google-research/google-research/master/slot_attention/object_discovery/train.py)
and [model implementation](https://raw.githubusercontent.com/google-research/google-research/master/slot_attention/model.py).
This is a matched HDF5 data/exposure budget candidate, **not** an official
500,000-step reproduction.

The CPU-only runner sets TensorFlow intra/inter-op threads to 2/1, limits
OMP/MKL/OpenBLAS/NumExpr threads, and uses `nice`. It is sequential by design:

```bash
bash collaborative_test/SW_0056_matched_slot_2500/run_seed.sh 0 dry-run
bash collaborative_test/SW_0056_matched_slot_2500/run_seed.sh 0 smoke
bash collaborative_test/SW_0056_matched_slot_2500/wait_for_low_load_and_run.sh --dry-run
bash collaborative_test/SW_0056_matched_slot_2500/wait_for_low_load_and_run.sh
```

The smoke mode performs a 16-image update followed by an 8-image update on 24
real training images. It exercises the auxiliary static-batch model, copies
current primary weights, applies its gradients to the primary variables, and
asserts finite loss plus a changed primary weight before completing. It does
not validate. Full runs write the seed, exact IDs/exposure plan, schedule,
model/trainer/checksum provenance, loss log, TensorFlow checkpoint, and
validation protocol into its seed directory. Valid completed phases can be
skipped; partial or invalid artifacts are never overwritten. There is no
parallel/GPU launcher. The full run then uses SW0046's predictor
and scorer on IDs 1320-1639 with protocol metadata identifying the scratch
checkpoint and training seed.

`run_seed.sh` is phase-aware. A training phase is skipped only if its completion
marker, `ckpt-1563` index/data, full protocol, and all 1,563 finite loss rows
validate. Its unique launcher log is created outside the output directory, then
moved to `training.log` only after successful validation. Training failures
retain the external log and partial output with a `FAILED` record containing
phase, return code, and UTC timestamp; partial training is never retried.
Inference is skipped only when its completion marker, exact IDs, predictions,
and protocol validate. Partial/invalid inference or scoring output is preserved
and stops recovery. `COMPLETED` is written only after training, inference, and
scoring artifacts all pass their full validators.

The separate sequential runner waits until **ten consecutive 60-second
samples** satisfy load1 <= 32, load5 <= 36, load15 <= 40, and
`MemAvailable >= 8 GiB`. It does no compute while load is high, validates the
24-image smoke first, then runs seeds 0, 1, and 2 sequentially. It checks the
same gate once immediately before each actual training, inference, or scoring
phase; concurrency is one. An exclusive lock, timestamped log, phase markers,
stale-PID verification, and no-restart policy protect a stopped/partial run.
After all three seeds pass, it writes an exclusive three-seed JSON and Markdown
summary with per-seed values, mean, and sample standard deviation for FG-ARI,
foreground IoU, and matched-object IoU. The summary requires the same
validation slice and protocol family across seeds and verifies GT-free
prediction provenance.

Pure NumPy tests check exact IDs, deterministic order, ten real appearances per
image, the 1,562+1 batch structure, and schedule metadata:

```bash
python -m pytest collaborative_test/SW_0056_matched_slot_2500/test_protocol.py -q
```

No training or evaluation was launched while preparing this experiment.

## CUDA seed1/2 acceleration

The original TensorFlow 2.15.1 environment is a CPU-only build. A separate
Python 3.10 environment at `envs/slot_attention_gpu_tf215` was therefore
created with `tensorflow[and-cuda]==2.15.1` and verified to report both
`is_built_with_cuda=True` and all four server GPUs. The completed/near-complete
CPU seed0 remains authoritative. Because CPU seeds1/2 were still near the start
of the fixed schedule, `queue_gpu_seed12.sh` can reproduce them from scratch in
separate immutable output directories when two allowed GPUs are idle.

The GPU runner preserves the exact model, seeds, ID permutations, 25,000 real
exposures, 1,563 updates, final batch of eight, learning-rate schedule,
validation slice, and scorer. It first runs the existing real 16+8-image smoke
on each GPU. CPU partial outputs are never modified. After both GPU seeds and
the CPU seed0 validate, a symlink-only summary root maps the three selected
outputs into the existing exclusive summarizer, which still checks the full
protocol family before writing the standard SW0056 summary consumed by SW0087.

## 2026-10-05 parallel execution update

Seed0 was already live under the guarded CPU scheduler. At the user's explicit
request to begin the matched-data comparison immediately, seeds1/2 were started
concurrently after verifying 48 logical CPUs, more than 80 GiB available memory,
and only two TensorFlow compute threads per run. This changes wall-clock
concurrency only: every seed retains its own deterministic order, 2,500 unique
images, ten exposures, 1,563 optimizer updates, checkpoint validation, and
aligned 320-image scoring. `wait_parallel_summary.sh` independently waits for
all three validated `COMPLETED` markers and then produces the exclusive summary.

If CPU seed0 remains the last bottleneck, `finish_gpu_all_seeds.sh` preserves
the two running GPU jobs and reproduces seed0 with the same validated CUDA
runner. It then validates and summarizes all three GPU outputs. Only the
execution backend changes; the data, exposures, update count, model, optimizer,
seeds, and evaluation contract remain fixed.
