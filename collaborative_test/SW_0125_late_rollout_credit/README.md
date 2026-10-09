# SW0125: late-rollout credit

This prospective recipe tests whether a 64-frame differentiable tail can
provide useful gradient credit while retaining the original 1024-frame S2Net
forward trajectory and the full 512-frame settled objective window. The prefix
is evaluated without gradients, but its phase and recurrent neuron states are
preserved; static graph and oscillator drive are prepared once live and reused
by the tail. This is truncated BPTT, not full-rollout BPTT.

`late_rollout.py` is the rollout helper. `verify_foundation.py` provides a
read-only actual-data check for one matched B16 training batch per seed; it
records zero optimizer updates and uses no GT. `run.py` registers a 4-batch
same-input preflight and 256-update, full three-seed pilot. `evaluate.py` uses
the unchanged fixed QCC evaluator, and `dispatcher.py` runs the nine stage tasks
only on exclusively leased idle GPUs without automatic retries.

The foundation probes passed actual 1024-frame trace/loss parity and RGB
gradient checks for all three seeds with zero optimizer updates. The full
pipeline passed nine local CPU/API tests (one additional CUDA test skipped).
Its four-batch GPU preflights passed for all three seeds; all named parameter
families changed in the disposable Adam checks. The full pilots started on
GPU 0, 1, and 3. Endpoint evaluations remain pending.
No performance improvement or all-gating contribution is claimed.


## Completed outcome (2026-10-09)

All nine registered stages passed execution validation. All three pilots completed
256 matched B16 updates; native fixed320 evaluations are complete. The long-forward
late64 candidate means (FG-ARI/foregroundIoU/objectIoU) are
0.767077/0.585208/0.592722, versus source97 0.776348/0.577070/0.601537
and matched short 0.767341/0.574580/0.594957. Every seed lost FG-ARI
relative to source97. Shared-image paired FG delta versus source is -0.009271
with 95% CI [-0.015291,-0.003216]; versus short it is -0.000264
with CI [-0.006622,0.006258]. Foreground coverage improved on average,
but object separation did not. This weakens the hypothesis that short-training
horizon alone explains the loss in this recipe; it does not isolate all temporal
mechanisms or establish gating usefulness or data scaling. Incumbent97 remains.
Close this recipe without coefficient/tail/threshold sweeps or 70k expansion.
Full records are in completed_results_20261009.json and three_seed_summary_20261009.json.
Earlier pending/running text is historical.
