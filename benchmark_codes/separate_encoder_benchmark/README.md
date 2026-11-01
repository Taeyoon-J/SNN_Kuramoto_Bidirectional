# SW0257: separated encoder training

This isolated experiment supersedes the encoder-through-dynamics SW0256 run.
The preserved same-seed SW0097 native16 core is mapped to native64 and remains
bitwise fixed, including the sensory projection and integration parameters.
All model outputs remain native64. The protected `vendor/benchmark` is unchanged.

18 conditions: model252/model253 x CLEVR/Tetrominoes/CLEVRTex Original x seed0/1/2.
Each encoder uses exactly 35,000 unique registered TRAIN images for 10 epochs,
physical batch16 with a final batch8: 350,000 exposures and 21,880 Adam updates.
No recurrent core forward/backward is executed during encoder training.

* 252 starts from the same-seed trained240 encoder. The existing direct
  gamma/effective-frequency teacher objective uses TRAIN masks. Base/context
  learning rates are 3e-5/3e-4. The initial teacher stays fixed. Unlike original240,
  the projection also stays fixed, as required by the current fixed-core policy.
* 253 starts from the same-seed SW0097 raw CNN encoder (224 parameters), trained
  by RGB autoencoder reconstruction MSE, Adam1e-3. Its auxiliary original
  CNNFeatureDecoder is initialized with seed257000+seed; it is never used for
  segmentation or inference. No ground-truth masks are opened during training.
* 252's existing trained neutral head is warm-started, then receives **3,500
  selected TRAIN images x10 additional epochs**, 35,000 single-image updates,
  Adam.003 with a new optimizer. Actual detached spike/membrane144 features and
  truth-free regions are cached once per image. CLEVRTex has no old native head,
  so it transfers the same-seed CLEVR head. Tetrominoes uses its existing adapted
  same-seed head. Every output group is an ordinary group; none is designated or
  deleted as background. All truth groups enter the existing CE/Dice matching.
* 253 retains the existing spike/component/geodesic readout, with no mask head.

Gradient norms are clipped to1. FP32, TF32 disabled; original data preprocessing,
preserved normalization, splits, predictions and protected evaluation are used.
Evaluation: fixed320 CLEVR (historical reserved set), fixed1000 Tetrominoes and
official TEST1000 CLEVRTex. Primary numbers are protected native-pixel scores;
historical CLEVR64-patch scores are also retained. Tet native64 predictions are
reduced by protected modal32 before lift to35 pixels.

Example (Linux, assets described by configs/frontier.json and source inventories):

```sh
python -B run_separate_encoder.py --output /new/isolated/output preflight
python -B run_separate_encoder.py --output /new/isolated/output queue --gpus 0,1,2,3
```

Each seed/dataset gets a separate two-update actual-CUDA admission and one-image
diagnostic evaluation before its full training starts. Admission weights never
initialize full training. Checkpoints preserve Adam/RNG/offset; code/data/source
bindings fail closed on changes. The final independent audit checks every TRAIN
ID/update, prediction hashes and recomputes all protected metrics. Failures keep
artifacts and are diagnosed before resuming; foreign GPU processes are untouched.

## Optional concurrency manager

`manage_separate_capacity.py --root OUTPUT` can fill a second worker slot per GPU
after a resource dry-run and actual CUDA memory profiling. The recorded original
13,000MiB allocator peak was below2,000MiB for representative252/253 diagnostics.
The additional worker reserves8,000MiB for scheduling, but keeps the ORIGINAL
13,000MiB allocator, identical scientific entrypoint, batch16 and training budgets.
The reduced8,000MiB allocator was **not adopted**: tiny252 encoder differences
propagated to detached head weights, despite identical diagnostic hard masks.
Raw comparison evidence is retained in `validation/concurrency_revision.json`.

The manager accounts for foreign GPU memory, adds1,024MiB GPU headroom, limits
each GPU to two scientific workers and reserves eventual10GiB host RAM per worker.
It only quiesces the coordinator in its verified20s sleep interval to serialize
launch bookkeeping; scientific workers are never signalled. It fails closed if
the locked coordinator source, profiled memory bound or source binding changes.
The original `queue` entrypoint remains usable without this optional manager.
