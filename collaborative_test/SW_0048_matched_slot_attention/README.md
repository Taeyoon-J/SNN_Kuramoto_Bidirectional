# SW0048 - Matched-data/pass-budget Slot Attention baseline candidate

This prepares three independent official Slot Attention object-discovery
models trained from scratch on the same HDF5 CLEVR training images (IDs
0-999) used by the SNN, with seed-specific initialization and data order.
Validation uses the aligned HDF5 IDs 1320-1639 through SW0046's full-image
predictor and shared patch scorer. The objective is to make the reference
better matched in data source and training exposure than the available
single pretrained transfer checkpoint.

The protocol follows Google's official object-discovery setup: model input
128x128, 11 slots, 3 Slot Attention iterations, reconstruction L2/MSE, Adam
at 4e-4 with epsilon 1e-8, and the official linear warmup followed by
exponential decay (`decay_rate=0.5`). See the [official training loop](https://raw.githubusercontent.com/google-research/google-research/master/slot_attention/object_discovery/train.py)
and [official model implementation](https://raw.githubusercontent.com/google-research/google-research/master/slot_attention/model.py).
This candidate is **not** a reproduction of the official 500,000-step run:
it uses our HDF5 split, full uncropped images, 11 inference/training slots,
batch 16, and 2,500 updates (40 image passes), with warmup/decay shortened to
250/2,500 steps.

Each pass independently permutes all 1,000 training rows. The shuffled pass
streams are concatenated before batching, so batch boundaries can cross a pass
boundary; no examples are dropped or under-sampled. This is preferable to
62 batches/epoch with eight dropped rows, and yields exactly 40 appearances per
training image over 2,500 updates. Images are read from the full HDF5 frame,
converted to float32 in [-1,1], and are not cropped. Targets/masks are never
loaded by the training program.

Each seed directory records `training_protocol.json`, `training_loss.csv`,
TensorFlow checkpoint files, and a completion marker. The full run then
predicts IDs 1320-1639 and scores them with the same SW0046 evaluation path.
Training is CPU-only because the designated TensorFlow environment sees no
usable GPU. Run scripts are prepared; no full training has been launched.

```bash
# A no-write plan check, or a two-update CPU smoke run on 32 images:
bash collaborative_test/SW_0048_matched_slot_attention/run_seed.sh 0 dry-run
bash collaborative_test/SW_0048_matched_slot_attention/run_seed.sh 0 smoke

# Full, sequential or concurrent CPU training for seeds 0, 1, and 2:
bash collaborative_test/SW_0048_matched_slot_attention/run_sequential.sh full
bash collaborative_test/SW_0048_matched_slot_attention/launch_parallel.sh full
```

The sequential launcher stops on the first failed seed. The parallel launcher
uses one log and output directory per seed and refuses pre-existing paths.
`run_seed.sh` also refuses to overwrite existing outputs. Smoke runs are
isolated under `*-smoke` and skip full validation.

Pure NumPy protocol checks (schedule, exact pass coverage, and deterministic
seed ordering):

```bash
python -m pytest collaborative_test/SW_0048_matched_slot_attention/test_protocol.py -q
```
