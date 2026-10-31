# SW_0125_late_rollout_credit

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## foundation_outcome_20261009.md

<details>
<summary>foundation_outcome_20261009.md</summary>

Source SHA256: `9db0cc8fa8f0abdf84281d8f392cd1dae57cffa9b2b790e143a124ab51954b0a`

````markdown
# SW0125 actual foundation result

All3 original same-seed SW0097 B16 TRAIN probes passed native1024 forward parity on2026-10-09. Every phase, component/mean spike/membrane trace and registered512-band phase/Q/analyticRGB forward loss matches exactly. All6 parameter families receive finite nonzero RGB credit through the last64 live steps. No optimizer updates, masks, or evaluation scores were used.

The firstGPUattempt failed only the averaged-membrane exact guard while everycomponent trace matched. It is preserved in original_foundation_failure_20261009.json and server results_archive/foundation_attempt_original_20261009/. The corrected helper uses native stack/permutation/reshape strides and unchanged strict equality guards. Corrected attempts and all3supervisor/childPIDs are terminal rc0; full reports and hashes are in completed_foundation_proofs_20261009.json.

Membrane gradient norms are6.38e-7/2.78e-7/1.40e-6; encoder norms .0367/.1378/.0417. Nonzero credit alone does not establish useful gate contribution, expected parameter changes, model improvement, or data scaling. Adam update and complete256-update training/evaluation remain pending; existing97 incumbent unchanged. Do not infer metric gains or full1024-step BPTT from this probe.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `9451ce5efa342bb24c5f6a8847756298b23e971ccf71a27d0c7487c6af5f25c1`

````markdown
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

````

</details>
