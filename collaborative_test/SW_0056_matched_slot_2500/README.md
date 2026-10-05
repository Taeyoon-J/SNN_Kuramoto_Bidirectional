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
bash collaborative_test/SW_0056_matched_slot_2500/run_sequential.sh full
```

The smoke mode performs a 16-image update followed by an 8-image update on 24
real training images. It exercises the auxiliary static-batch model, copies
current primary weights, applies its gradients to the primary variables, and
asserts finite loss plus a changed primary weight before completing. It does
not validate. Full runs write the seed, exact IDs/exposure plan, schedule,
model/trainer/checksum provenance, loss log, TensorFlow checkpoint, and
validation protocol into a fresh seed directory. Existing outputs are refused;
there is no parallel/GPU launcher. The full run then uses SW0046's predictor
and scorer on IDs 1320-1639 with protocol metadata identifying the scratch
checkpoint and training seed.

Pure NumPy tests check exact IDs, deterministic order, ten real appearances per
image, the 1,562+1 batch structure, and schedule metadata:

```bash
python -m pytest collaborative_test/SW_0056_matched_slot_2500/test_protocol.py -q
```

No training or evaluation was launched while preparing this experiment.
