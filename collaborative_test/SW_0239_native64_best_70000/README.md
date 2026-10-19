# SW0239 native64 best-model TRAIN70,000

Continue the highest independently audited full320 candidate among SW0227/SW0237,
using all 70,000 distinct registered TRAIN images for ten complete passes. Keep the
original mask-supervised source-frequency teacher, gamma anchor, actual sensory
projection learning and spike readout. Teacher computations stay on CPU, matching
the accepted target cache exactly. Adam resets for this new data stage.

One supervisor leases at most one GPU and runs seed0, seed1, seed2 sequentially.
Other experiments may use other GPUs. It yields only its own worker when a foreign
GPU user appears; the last committed Adam/model/RNG boundary supports resumption.

Tests, from the repository root:

```sh
python -B -m pytest -q --import-mode=importlib -p no:cacheprovider -W ignore::pytest.PytestConfigWarning collaborative_test/SW_0239_native64_best_70000
```

After source verification, select once, complete all three CPU admissions, then
start exactly one supervisor:

```sh
python -B -m collaborative_test.SW_0239_native64_best_70000.train --select
python -B -m collaborative_test.SW_0239_native64_best_70000.train --admit --seed 0
python -B -m collaborative_test.SW_0239_native64_best_70000.train --admit --seed 1
python -B -m collaborative_test.SW_0239_native64_best_70000.train --admit --seed 2
python -B -m collaborative_test.SW_0239_native64_best_70000.queue
```

The worker checks a real batch's GPU gradients and one actual native64 TRAIN
rollout before optimizer updates. Checkpoints commit every 32 updates and at every
complete pass. Up to 31 uncommitted updates may be replayed after interruption.

Each final seed predicts the fixed 320 validation images. Scoring reads evaluation
GT only after all three prediction protocols and comparable Slot artifacts have
validated. Report all three metrics and the seed mean, preserving the distinction
between mask-supervised ours and reconstruction-unsupervised Slot. Visualizations
use predetermined images 1320/1321/1322 and seed0 for both models.

Source/data preflight records alone do not establish queue admission or training.
Runtime artifacts belong in the generated artifact tree through results_archive.
