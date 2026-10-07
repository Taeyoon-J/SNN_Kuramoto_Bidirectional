# SW0093: why the current recipe weakens during 70,000-image training

2026-10-06. User requested renewed analysis and improvement toward beating Slot
Attention on 70,000 images. This record analyzes completed runs; no new GPU
training has been launched and no training/evaluation defaults have changed.

## Verified comparison

Same held-out HDF5 IDs1320–1639, 320 images, patch16x16, three seeds:

| Model / checkpoint | FG-ARI | Foreground IoU | Matched-object IoU |
|---|---:|---:|---:|
| Our model, 70k, epoch1 | .774323 | .548028 | .603848 |
| Our model, 70k, epoch3 | .752242 | .548378 | .614339 |
| Our model, 70k, epoch10 | .677605 | .469147 | .561948 |
| Slot Attention, same70k, epoch10 | .774933 | .203589 | .206937 |

We lost FG-ARI at the matched ten-pass endpoint. Our epoch1 and Slot epoch10
differ by only .000610; this numerical difference does not establish a reliable
performance difference. Their training budgets differ. The released Slot
checkpoint's .894641 is a separate pretrained transfer reference, not a
matched-budget three-seed baseline. A final generalization claim will need a
separate untouched test split after validation-driven model selection.

## Confirmed mechanism mismatch: opposite spikes mean different things

The training helper `training/train_s2net_core.py:_component_spike_synchrony`
multiplies `loss_function.py:signal_synchrony` across components. That function
uses **absolute** centered correlation. The actual classifier in
`spike_classifier.py:spike_synchrony_affinity` multiplies correlations after
clipping negative values to zero. Thus:

- Training affinity: product(abs(correlation)).
- Classifier affinity: product(max(correlation, 0)).

Four perfectly anticorrelated component traces give training affinity1 and
classifier affinity0. This is a semantic mismatch in the current training
recipe, not just a difference in naming. The current loss can reward strong
opposition as strong similarity although prediction discards those edges.
It was not identified in the earlier scaling report and should have been
checked before assuming longer training would help.

This counterexample proves the mismatch, not that opposite spikes caused the
observed score decline. Measure signed correlations and the resulting edge
disagreement on real checkpoints before attributing the full decline to it.

## Observed failure, rather than speculation about more data

All three SW0090 seeds finish with lower training loss and lower held-out
FG-ARI than at epoch1:

| Seed | Loss epoch1 -> epoch10 | FG-ARI epoch1 -> epoch10 | Mean predicted objects epoch1 -> epoch10 |
|---|---|---|---|
| 0 | 19.818898 -> 18.112345 | .818470 -> .783340 | 6.87 -> 9.65 |
| 1 | 19.887645 -> 15.975247 | .746697 -> .634487 | 7.47 -> 7.16 |
| 2 | 26.392057 -> 24.807842 | .757803 -> .614987 | 6.93 -> 10.41 |

Loss is not monotonically decreasing for every seed. Seed2 is especially
unstable after epoch3. Per-image FG-ARI declines on 192/320, 252/320 and
262/320 images respectively. This is broader than a few bad examples.

GT mean count is6.196875. Three-seed mean count MAE rises from1.7760 to3.2792;
exact count accuracy falls from19.79% to10.73%. Two seeds show substantial
oversegmentation. Seed1's count MAE stays close to2 while FG-ARI deteriorates,
so a count-only solution would leave incorrect grouping unresolved.

The active objective consists of bimodality, density, variance barrier and
spatial smoothness on phase and spike affinity matrices. It has no direct
image reconstruction or instance assignment objective in this recipe. Density
target .867 is shared across images. These global statistics do not specify
which patches belong together. Spatial coherence smooths each patch's mean
affinity to other patches, rather than supervising pairwise object boundaries.
Better objective values therefore need not imply better object grouping.

## Additional confounds and hypotheses

1. **Epochs hid a 27.9x increase in optimizer updates.** At batch16, 2,500
   images x10 passes give1,570 updates; 70,000 x1 gives4,375; 70,000 x10
   gives43,750. Adam uses constant lr .0003 in this trainer. The current
   evidence shows degradation with continued optimization on the same70k;
   it does not isolate a harmful effect of the number of unique images.
2. **The whole best core was not continued.** `load_graph_checkpoint` loads
   only `graph_generator.*` and freezes it; downstream parameters are freshly
   initialized per seed. The old SW0072 seed0 also reused a previously
   graph-trained run, while its other seeds froze that graph. The small-vs-large
   comparison is not a fully matched initialization control.
3. **Temporal windows differ.** Training simulates64 steps and uses the last32;
   evaluation simulates1024 and uses the last512. Longer-run behavior is not
   directly trained by the short window. Its causal effect is unmeasured.
4. **Feature adaptation is limited.** Cached encoder outputs and the graph are
   frozen. More images cannot improve those representations in this recipe.
   This is a possible scaling ceiling, not evidence that unfreezing everything
   will help. SW0073–76 and SW0088–89 already found negative or tradeoff results
   with naive reconstruction, encoder/graph adaptation and preservation losses.

## Prioritized next experiments and decision rules

1. **Checkpoint diagnosis without retraining.** Use epoch1/3/10, the same
   fixed image subset, to compare training absolute affinity and actual positive
   classifier affinity at short/long temporal windows. Measure edge
   disagreement, GT same/different-object separation, constant trace fraction,
   connected-component fragmentation and stage-wise signal/gradient flow.
   GT is diagnostic only and never enters prediction or unsupervised training.
   This discriminates sign mismatch, temporal drift and downstream degradation.
2. **One controlled objective change.** Make spike-loss affinity match the
   classifier's positive-correlation product, keeping initialization, data,
   update budget and evaluator fixed. Preserve the old objective as an explicit
   control. Verify finite, nonzero gradients before GPU training: clipping
   negative correlations can remove gradients, so a successful forward check
   alone is insufficient. First evaluate after a short fixed update budget;
   stop this direction if grouping quality or gradients are worse.
3. **Separate unique data from update count.** Compare 2,500 and70,000 unique
   training pools at the same number of updates, identical frozen graph and
   downstream initialization. Count how many unique images the70k arm actually
   sees; a short run cannot be called a complete70k training. Separately continue
   the full best core checkpoint with lower LR/decay and keep fresh-core training
   as its own control. An LR improvement protects existing performance but is
   not by itself evidence that more data creates useful representations.
4. **Train actual object grouping if alignment alone is insufficient.** Build
   a differentiable assignment path driven by spike/gate affinity, with
   competition between groups and an image/feature explanation objective.
   Ensure rhythm is necessary to the assignment and reconstruction cannot
   bypass it. Compare against rhythm removal and component/gate ablations;
   measure retained same-object separation through Kuramoto, sinusoidal,
   dendritic, membrane and spikes. This is an architectural candidate, not a
   proven fix; it differs from merely increasing failed reconstruction weights.

Only promising short-budget arms should expand to full70k and three seeds.
Keep the existing data and fixed evaluator. Select checkpoints on validation,
then compare final models against Slot using three-seed means and independent
test data; report all three mask metrics and count accuracy. Beating .774933 is
the current matched70k FG-ARI target; beating the released .894641 reference is
a distinct, harder target. No outcome is guaranteed.

## Reproduce

`python collaborative_test/SW_0093_scaling_failure_analysis/analyze.py`

Only the standard library is needed. The script checks image IDs, all ten
training epochs per seed, absence of GT prediction, finite per-image FG-ARI,
and agreement between per-image means and reported metrics. It writes
`results/diagnosis.json`. Archived completed training logs are in `results/`.
