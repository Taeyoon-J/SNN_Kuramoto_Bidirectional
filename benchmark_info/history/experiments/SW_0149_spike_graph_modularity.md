# SW_0149_spike_graph_modularity

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## design_review.md

<details>
<summary>design_review.md</summary>

Source SHA256: `7fdb8da0059db19fd13448877596c78a8e97c64b0815b747396cd8df5b4e315f`

````markdown
# SW0149 objective transition after SW0148

Design decision, not an executed training result. SW0148 completed with seed1 FG-ARI source .754135, encoder/graph .745394, and joint .733767. Joint minus source and joint minus control paired FG confidence intervals were negative. Fixed320 error analysis found increased foreground splitting, merging, and missed foreground. On the same first64 TRAIN images, count-weighted boundary MSE was source .055386, control .055014, and joint .055619. Improving selected RGB-edge strata did not establish better object grouping.

Close the unchanged seam-loss recipe and hold SW0147 training under that recipe. Preserve its independently tested efficacy kernel for a future justified mechanism intervention. Preserve all completed artifacts and the native32 incumbent.

The next candidate replaces the objective with degree-corrected modularity of the actual-spike Q relation. A frozen source encoder/graph supplies a nonnegative symmetric adjacency A with zero diagonal. For each image, d=A1, Z=sum(d)>0, M=A-dd^T/Z, and L=-sum(M*Q)/Z. Average losses across images. Q is the unchanged production positive-clamped component-product spike affinity, with diagonal one. The source graph is a fixed reference, not semantic ground truth. No masks, RGB seam targets, decoder, source-Q distillation, or prescribed affinity density enter training.

Compare encoder/graph training with encoder/graph plus native core and integration scalar training from identical original source97 seed1 states. Keep native32, ordered4096 TRAIN IDs, 256 logical B16/microB4 updates, T1024/settle512/live-tail64, existing learning rates, clip and scalar projection. Compute frozen source graph per batch without another temporal rollout or a large adjacency cache. Never initialize from the failed SW0148 trained weights.

Before training, verify actual native32 source parity, finite positive total degree, positive and negative off-diagonal modularity credit on the fixed first64 TRAIN inputs, and two disposable real optimizer updates. Check registered parameter credit and actual changes without forcing small scalar gradients to be large. Preserve strict checkpoint lineage, Adam, RNG and runtime restoration across interruptions. CPU fixture success alone does not admit training.

Freeze both fixed320 prediction archives before opening targets. Report source, control, joint and comparable Slot32 metrics and paired image intervals. Expand only if joint FG improves by at least .01 over both source and control, both paired95% FG interval lower bounds are positive, and both IoUs exceed registered Slot32 three-seed means by .05. Otherwise close the fixed recipe without a coefficient, threshold or density sweep.

Success would justify seed0/2 replication, then separate fresh native32 70000-image and scaling tests. It would not establish all-gate necessity: explicit causal interventions remain required. Source graph errors and a mismatch between modularity and object grouping remain substantive risks to be measured at the endpoint.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `5d8d0219741f1a1d7475194afd211bd53edc4402dd22f84648641c033225529b`

````markdown
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

````

</details>
