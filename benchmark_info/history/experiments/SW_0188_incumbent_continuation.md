# SW_0188_incumbent_continuation

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `d2581327631d5e3fbf7a2ddd67adbfa5e4618464deb0c67d5af70a960a25b457`

````markdown
# SW0188: continue the actual SW0186 incumbent weights

SW0187's joint-encoder unique1024 arm did not improve FG over its contemporaneous
repeated512 control. Close that scaling recipe. This experiment tests the
previously requested alternative: retain the complete successful incumbent
instead of returning to Source097 for every new training stage.

Initialize each seed0/1/2 from its matching completed SW0186 checkpoint, including
the trained original RGB CNN, context, graph, rhythm/core and integration tensors.
Verify the entire model digest equals that incumbent's final digest on CPU and
again on GPU. Reset all five Adam groups to empty state explicitly. This is
weight continuation with a new optimizer stage, not exact Adam/RNG resumption.
Do not reset the context or any gating to its fresh Source097 values.

Keep the successful QCE objective, temperature .1, all five learning rates,
TRAIN512 twice, fixed64 proposals/B16/micro4, T1024/settle512/live-tail64,
clip1, projections and strict measured full-forward loss accept/rollback.
No additional data or gates are introduced in this test. Its purpose is whether
an additional matched adaptation budget improves the actual incumbent.
The incumbent itself already inherited full70k RGB core training; these image
counts describe the additional supervised native32 stage only.

Before deployment: compile every source, test queue/rollback/readout/scoring,
exercise the actual score producer ID at the audit entry point, and verify
independent arithmetic against actual archived score vectors. Before CUDA:
validate all three original SW0186 checkpoints/manifests, exact warm model SHA,
trained original CNN SHA, frozen preprocessing and five-family empty Adam.
First two real GPU proposals must demonstrate emitted-S and all-family credit,
full-forward loss parity and a committed actual model/CNN/Adam update.

Use the existing fixed320/native32 modal contract and unchanged spike-derived
Q/QCC/SW165 production assignment. Freeze all three seed predictions and both
conditional/standalone endpoints before reading validation GT for the new score.
Primary gate: standalone FG mean strictly above SW0186 and paired-image95% CI
lower above zero, plus all three metric means strictly above comparable Slot70k.
Report per-seed scores, both endpoints, both IoUs and FG.87 separately. An
independent audit must recompute actual masks, modal GT, seed means, intervals
and gates. Never declare unsupervised improvement or gating causality from this
supervised continuation. Native32 joint-encoder70k remains a separate goal.

One fixed experiment; no acceptance-count retry or small-hyperparameter rescue.
If the improvement gate fails, close this continuation recipe. Queue only on
exclusive idle GPUs and yield our worker when a foreign user arrives.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `b1d2848e41d3cfe56c143f2210d20e4ac651739a72ab835d80f2f2c3bf1f26c0`

````markdown
# SW0188 — incumbent weight continuation

Preparation in progress; no new GPU training or performance claim yet.
The complete matching-seed SW0186 weights are retained, including its learned
RGB CNN and context. Five-group Adam is explicitly reset. Additional TRAIN512
twice/64 proposals, QCE and production masks remain unchanged. This tests
continued learning from the actual incumbent, not new data scaling.

See [registered design](DESIGN_KO.md). Evaluation compares fixed320/native32
three-seed results directly with SW0186 and comparable Slot70k. No .87,
unsupervised, full native32 joint70k or gate-causality claim is made.

Registered and running: all eight supervisors verified live; seed0/1 actual GPU workers started, seed2 is queued. Local/server39 tests, six actual CLIs, all three exact incumbent tensor digests, trained CNN hashes, five-family empty Adam and Slot archive admission passed. The first seed0 actual GPU update has already passed full-forward parity and changed the trained CNN. First-two admission and full320 scores remain pending. Peer patch_v2 was checked this cycle and remains cab6c676. Receipts: results_archive/cpu_v1 and start_v1.

Seed0 actual GPU credit/parity, accepted Adam steps and changed original CNN
were independently inspected; see `results_archive/start_v1/seed0_first_two_GPU_admission.json`.
This admission does not establish improved masks or a completed performance result.

Seed1 actual GPU credit/parity, accepted Adam steps and changed original CNN
were independently inspected; see `results_archive/start_v1/seed1_first_two_GPU_admission.json`.
This admission does not establish improved masks or a completed performance result.

Seed0 training completed and its actual checkpoint/Adam/CNN and matching SW0186 start were validated:64 proposals,33 accepted; metadata are in results_archive/completed_seed0_v1. Seed2 has started on GPU2. Seed1 yielded GPU3 to a foreign process with its11-proposal/6-accepted checkpoint preserved and the same supervisor waiting to resume. Predictions/scoring remain queued. No completed fixed320 score or incumbent change is claimed. Model sources are unchanged.

Seed2 actual GPU credit/parity, accepted Adam steps and changed original CNN
were independently inspected; see `results_archive/start_v1/seed2_first_two_GPU_admission.json`.
This admission does not establish improved masks or a completed performance result.

Seed2 training is complete and validated (64 proposals,25 accepted), preserving the exact SW0186 seed2 start and actual CNN/Adam learning. Seed0 prediction has started. Seed1 was fresh-requeued only for resources with identical queue fingerprint; its old11 completed proposals/6 accepted updates and full partial files are preserved and excluded from delivered weights. New parent1663660 and original prediction parent1614111 were verified live. Current-run admission remains0/2 until replacement seed1 starts. See [resource amendment](RESOURCE_REQUEUE_KO.md) and results_archive/resource_requeue_seed1_v1. An early registration observation lacked the fingerprint field; the same replacement PID was verified later, with no duplicate launch. Model sources remain unchanged.

Seed0/2 both-endpoint fixed320 predictions are frozen and validated in frozen_seed0_v1/frozen_seed2_v1. Replacement seed1 has actually started on GPU2; its first two proposals passed emitted-S/five-family credit and full-forward parity, with committed CNN/Adam learning. Its distinct receipt is start_v1/seed1_resource_requeue_first_two_GPU_admission.json. All current delivered runs now have GPU admission; the original excluded seed1 proof remains historical. No SW0188 GT score is available yet.

Completed: all three seeds and full fixed320 audit passed, but the FG improvement gate failed. Means .831401/.559170/.681908 versus SW0186 .834681/.569040/.689802. Close this continuation recipe; incumbent unchanged. See [results and limits](RESULT_KO.md). Source imagewise scores remain private; aggregate score/audit are shared.

````

</details>

## RESOURCE_REQUEUE_KO.md

<details>
<summary>RESOURCE_REQUEUE_KO.md</summary>

Source SHA256: `d3f655c68478a0e0cef3db4bf1b0661747d3c862146c0df785760f947f49fa05`

````markdown
# Resource-only amendment: seed1 fresh requeue

Registered before execution and before any SW0188 validation scoring.
Seeds0/2 have completed training. Seed1 yielded GPU3 to a foreign job and its
unchanged supervisor has waited for that physical device for over an hour.
The same-runtime checkpoint contract binds that partial run to GPU3, so it
cannot use GPU2 after the other queued work releases it.

Fresh-requeue only seed1 from the exact same completed SW0186 seed1 weights and
empty five-group Adam, retaining the original code, dataset/order, loss, RNG
proposal schedule, learning rates,64-proposal budget and queue fingerprint.
The old11 completed proposals/6 accepted updates and full checkpoint/log/state
files will be preserved under the private runtime archive. Those11 proposals
are excluded from delivered model weights and reported as additional compute.
This is a resource-driven resubmission, not a failed-model retry or a choice
based on validation scores. No SW0188 score or validation mask selects it.

Delivered models still use64 proposals each (192 across three seeds); the old11
completed proposals are separate overhead, for203 completed proposals if all
three delivered runs finish. Do not count preserved partial updates as new
training exposures in the delivered seed1 model.

The read-only dry run verified exact current PID/argv/start ticks and our UID,
no child worker, foreign ownership on GPU3, the preserved11-proposal boundary,
all original model/test/DESIGN source hashes and the same CPU/queue identity.
Operational README edits are excluded from immutable model-source checks.
Old queue fingerprint: `4921773107d51bcfa9a4ebc674513514c90a596724071940d42de7f40886eafe`.

Execution briefly pauses only our waiting seed1 prediction parent, stops only
our waiting training parent, verifies the lock is released, moves the partial
files within the checked research workspace without deleting them, registers
one new parent and resumes the same prediction parent. Original launch and GPU
admission receipts remain historical evidence; the replacement needs a fresh
first-two-GPU admission. Never signal or share the foreign process.

Models0/2 and all other queues remain unchanged. A fresh queue may use any
exclusively idle GPU under the existing admission policy. If an observation
times out, inspect the same replacement PID and journal; do not launch again.

The replacement actually started on GPU2 and passed its own first-two admission. Identical starting tensor and proposal RNG hashes do not guarantee bitwise-identical optimizer/rollout trajectories across physical GPUs; no such equivalence is claimed. Selection remains resource-only and precedes SW0188 GT scoring.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `6b0246cea2b2ca91ed423b68ea48fca45d46c523ef76d088aff2395414ce4f2b`

````markdown
# SW0188: incumbent continuation did not improve FG

All three training runs, both fixed320/native32 prediction endpoints and the
independent actual-mask audit are complete. Retaining the complete SW0186 weights
and adding another supervised TRAIN512 twice/64-proposal stage with reset Adam
does not pass the improvement gate. Close this exact continuation recipe.

| Model | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW0188 continuation | .831401 | .559170 | .681908 |
| SW0186 incumbent | .834681 | .569040 | .689802 |
| Comparable Slot70k epoch10 | .781652 | .204360 | .203165 |

| Seed | FG-ARI | Foreground IoU | Object IoU | New accepted updates |
|---|---:|---:|---:|---:|
| 0 | .835862 | .790242 | .680340 | 33 |
| 1 | .813989 | .432728 | .686333 | 28 |
| 2 | .844351 | .454540 | .679052 | 25 |

Against SW0186, FG changes by -.003280, paired-image95% CI
[-.009268,.002801]. This establishes neither an FG improvement nor a definite
FG decrease. Foreground IoU changes by -.009870, CI[-.019725,-.000340];
Object IoU changes by -.007894, CI[-.013903,-.001883]. Only seed2 improves FG.
All three overall means still exceed comparable Slot under the fixed contract,
but no incumbent change, .87 result or benefit of extra optimization is claimed.

Conditional means are .799926/.609975/.653802. Both endpoints and all three seeds
were frozen before the new GT score. The independent audit recomputed actual
1,920 candidate endpoint/image rows and960 Slot rows, modal GT, means, confidence
intervals, source encoder learning and gates. Local archival also bound all
three actual training manifests, frozen protocols and original SW0186 starts.

All three source/CNN/Adam stage bindings passed: each delivered run starts exactly
from its same-seed completed SW0186 weights, not fresh Source097. The five-group
Adam reset is explicit. Its effect is not separated from the extra64 proposals.
The old resource-interrupted seed1 partial (11 proposals,6 accepted) is preserved
and excluded from delivered weights. Delivered runs total192 proposals, plus11
completed proposals of disclosed overhead. Replacement seed1 passed its own
GPU admission. No SW0188 validation score selected the replacement.

These are extra supervised native32 adaptations; the inherited Source097 core
already saw the full70k RGB pool. A new native32 joint-encoder70k pass remains
outstanding. Comparison uses the same evaluation contract, not identical model
training targets/budgets. Unsup improvement, gate causality and optimal gamma
ordering are not established. SW0165 remains the separate unsupervised anchor.

SW0187 data expansion and this simple continuation both failed to improve FG.
SW0190 separately rejects two oracle-count spectral readouts on TRAIN16.
Keep the successful production partition and investigate a training objective
that responds to actual partition errors; do not rescue these recipes with
another small learning-rate, temperature or clustering sweep.

Score SHA256: `f7899e1215e3bb4cd341184ceb941663d221b3a1e0bcfe9d12f5577475375490`.
Audit SHA256: `f9bb6b2d8e825291f683b6b6cec52f8a672ed95a35ff621f88400ee90b5c4d93`.
[Public aggregate score](results_archive/score_v1/summary.json),
[independent audit](results_archive/score_v1/completion_audit.json),
[archive receipt](results_archive/score_v1/archive_receipt.json).
The imagewise score is preserved privately and excluded from Git publication.

````

</details>

## results_archive/cpu_v1/registered_README.md

<details>
<summary>registered_README.md</summary>

Source SHA256: `4728eb1fd494925ef6069be03fed974d411d9519ce2afb30c2e7eb4725db43c4`

````markdown
# SW0188 — incumbent weight continuation

Preparation in progress; no new GPU training or performance claim yet.
The complete matching-seed SW0186 weights are retained, including its learned
RGB CNN and context. Five-group Adam is explicitly reset. Additional TRAIN512
twice/64 proposals, QCE and production masks remain unchanged. This tests
continued learning from the actual incumbent, not new data scaling.

See [registered design](DESIGN_KO.md). Evaluation compares fixed320/native32
three-seed results directly with SW0186 and comparable Slot70k. No .87,
unsupervised, full native32 joint70k or gate-causality claim is made.

````

</details>
