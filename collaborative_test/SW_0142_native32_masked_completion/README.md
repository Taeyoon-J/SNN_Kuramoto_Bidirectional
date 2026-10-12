# SW0142 native32 masked completion

SW0142 tests whether a fixed native32 spike-slot model improves through masked-image completion, comparing a frozen-core `encoder_graph` arm with a `core_trainable` arm. The source is the already-trained SW0097 seed1 model, mapped through the frozen SW0135 native32 stack. It has prior 70,000-image training at 16×16 resolution; this pilot is not fresh native32 pretraining and does not establish a scaling result.

The nine-stage queue is: write the 4,096-image float16 masked spike cache; warm the shared binder/decoder for 256 B16 updates; run one disposable B16 preflight per arm; train each arm for 256 B16 updates; predict the same fixed 320 images per arm; then score on CPU. Cache/warm/training/prediction do not read ground truth. The scorer validates and freezes prediction/reference hashes before loading the 320 targets. It reports the primary native32 P-derived hard-slot masks and the unchanged QCC readout. The registered gate uses 10,000 paired-image bootstrap samples with `RandomState(139)` and common resample indices.

The protocol also records first-64-TRAIN reconstruction diagnostics for the original assignments, row-permuted P, and within-coarse-RGB-bin permuted P. These diagnostics are descriptive, with no pass threshold. A reconstruction or nonzero-gradient result alone does not show object segmentation or gate necessity. Failure closes this recipe; it does not authorize sweeps or automatic seed expansion.

From the repository root, run the CPU tests with CUDA hidden:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m unittest collaborative_test.SW_0142_native32_masked_completion.test_run collaborative_test.SW_0142_native32_masked_completion.test_queue -v
```

The registered CLI is `run.py --stage {cache,warm,preflight,train,predict,score}`. `stage_queue.py --dry-run` displays the fixed stage plan without creating outputs. Running the queue creates artifacts and requires the reviewed owner-aware GPU lease environment; preserve existing outputs and do not restart an interrupted stage automatically.
