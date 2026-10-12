# SW0143: native32 selective-geodesic screen

This is a frozen-inference comparison on the same 320 validation images for source-native geodesic reference **R**, fixed effective contrast 8 with the original linear cosine cost **C**, and the same contrast with squared cosine dissimilarity **T**. All arms use the unchanged native actual-spike QCC readout at 32×32; predictions are made without target masks. R is replayed from the SW0097 source checkpoint and must match the existing SW0137 labels exactly for all 320 images before the intervention arms proceed.

The seed preflight uses four source-ordered TRAIN examples, no targets, and no optimizer updates. It requires exact SW0135-versus-R trace parity separately at B1 and B4. B1/B4 numeric differences are recorded for context only; changing batch size is not a rejection criterion. Both C and T must also complete finite B4 rollouts. The queue runs preflight, then the three R replays, then C/T predictions, and opens target masks only in the final CPU scoring task after validating every prediction and the frozen Slot32 reference.

Scoring uses the production 4×4 modal pooling from native 128×128 targets to 32×32 and reports foreground ARI, foreground IoU, and matched object IoU by seed and as three-seed means. Paired intervals use 10,000 shared-image bootstrap draws with `RandomState(139)`. The registered gate is reported as separate conditions; this screen does not establish fresh native32 training, scaling, gate necessity, or unconditional superiority.

Run the local CPU contract tests from the repository root with CUDA hidden:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -c "import numpy; import unittest; s=unittest.defaultTestLoader.discover('collaborative_test/SW_0143_native32_selective_geodesic', pattern='test_*.py'); r=unittest.TextTestRunner(verbosity=2).run(s); raise SystemExit(not r.wasSuccessful())"
```

Inspect the task plan without creating state or reserving GPUs:

```powershell
python collaborative_test/SW_0143_native32_selective_geodesic/geodesic_queue.py --dry-run
```

Queue execution is owner-aware and uses exclusive shared GPU leases. It does not retry failed stages or terminate foreign processes. Outputs are create-only under `results_archive/`.
