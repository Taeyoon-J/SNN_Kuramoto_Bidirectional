# SW0149: native32 source-graph modularity

This candidate trains the native32 encoder/graph (and, in the joint arm, the
remaining native core and twelve integration scalars) to increase degree-
corrected modularity between the frozen SW0097 seed1 source graph and the
production actual-spike affinity matrix. The objective fixes Q's diagonal to
the self-similarity one while leaving every off-diagonal production Q value
unchanged; the classifier/readout itself is not modified. The graph is a fixed reference, not
semantic ground truth. No masks, RGB seam loss, decoder, or prescribed graph
density enter the objective.

The preflight first validates the saved SW0135 exact native32 parity receipt,
then freshly compares mapped-source graph and actual-spike Q against the
native source on the first registered B4 TRAIN images. It also checks both
positive and negative off-diagonal modularity entries over the first 64 TRAIN
images and performs two disposable logical updates for each arm. It is
CUDA-only, reads registered TRAIN RGB only, never opens ground truth, and
always reports `training_admission: false`; it is not a training run.

From the repository root, the registered preflight command is:

```powershell
python -m collaborative_test.SW_0149_spike_graph_modularity.preflight `
  --device cuda:0 `
  --output collaborative_test/SW_0149_spike_graph_modularity/results_archive/preflight_seed1.json
```

The output path is create-only. Kernel and receipt-contract CPU tests can be
run with CUDA hidden:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m unittest collaborative_test.SW_0149_spike_graph_modularity.test_modularity collaborative_test.SW_0149_spike_graph_modularity.test_preflight
```

The one-task owner-aware GPU queue can be inspected without starting it:

```powershell
python -m collaborative_test.SW_0149_spike_graph_modularity.preflight_queue --dry-run
```

After the source preflight report has been completed and validated, the
two-arm pilot queue can also be inspected or started from the repository root:

```powershell
python -m collaborative_test.SW_0149_spike_graph_modularity.stage_queue --dry-run
python -m collaborative_test.SW_0149_spike_graph_modularity.stage_queue
```

The queue runs both 256-update arms on separate owner-checked GPUs, then each
arm's fixed320 prediction, and finally scoring in a CUDA-hidden CPU process.
It never overwrites existing outputs. GPU-owner interruption may resume a
training arm only from its validated logical-update checkpoint; predictions
retry only while their final output is absent. Ground truth is opened by the
last scoring stage, after both candidate predictions and registered references
have passed their hash and ID checks. The score reports the preregistered
paired bootstrap and promotion gates; it does not expand training automatically.

Passing the CPU tests or preflight does not admit training or establish a
semantic interpretation of the source graph. Fixed pilot expansion and
promotion criteria are recorded in `protocol.json`.
