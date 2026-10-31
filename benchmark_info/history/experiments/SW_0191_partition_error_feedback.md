# SW_0191_partition_error_feedback

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `601645c6e2b6d08fe925f6dfb4ddc3eff7bd92ce1301a9f8a5cc54ff1745c597`

````markdown
# SW0191: production-partition error feedback

SW0188's extra QCE adaptation did not improve the exact SW0186 incumbent.
SW0190's two oracle-count spectral readouts are worse in all three TRAIN seeds.
Retain the successful production QCC/geodesic partition and test feedback from
its actual mistakes, rather than replacing the readout or repeating SW0185's
local-neighbor margin loss.

Initialize all three source seeds from the exact completed SW0186 weights,
including RGB CNN/context/graph/core/integration, with explicitly empty five-group
Adam. Keep TRAIN512 twice/64 proposals, B16/micro4, all learning rates,
T1024/settle512/live-tail64, projections and clip1. Inference is unchanged.
This is supervised extra adaptation, not fresh native32 joint70k training.

Before each proposal, use a same-RNG no-gradient native rollout and the actual
production QCC/foreground-induced merge to mine predicted labels on that TRAIN
batch. Freeze those labels for the gradient forward and the post-proposal
full-forward loss. Never re-mine targets during the acceptance comparison.
This avoids changing the objective when the discrete partition changes.
Record the frozen context SHA. All64 gradient losses must match their same-RNG
pre-proposal full-forward composite losses within the existing tolerance.

Add coefficient1 feedback to unchanged product-QCE temperature.1:
same-GT/different-predicted-label pairs receive mean(1-signed_component_rho)^2
over four components, class-balanced over affected GT classes. Different-GT/
same-predicted-label pairs receive production Q^2, balanced over unordered GT
class pairs. Use each unordered patch pair once, excluding self pairs. Average
the positive/negative terms that exist; return a connected zero when no error
pairs exist. Background label0 participates as a cluster. Accumulate large group
sums in float64 for stable strict-loss checks. This is a stop-gradient mined
contrastive surrogate, not a differentiable ARI or a structured-inference bound.

Actual source RGB/normalization/graph paths and all four gating-to-spike routes
remain intact. First two GPU proposals must prove finite-positive total spike
gradient, all five family gradient/proposed delta credit, full-forward parity,
and separately measured feedback-only emitted-S and trainable-parameter gradient
RSS. Report its per-family parameter credit without requiring every auxiliary
family to have nonzero credit. This prevents a gradient into detached-prefix
spike outputs alone from being called effective model learning. A dead auxiliary
is a scientific rejection, not evidence that the new loss worked. Learning may
help fragmentation/merging; this alone does not prove gate causality or solve
fully silent foreground or automatic object counting.

Exclusive GPU ownership and yielding remain mandatory. To prevent a yielded
checkpoint waiting indefinitely for its old physical index, SW0191 stores a
compatibility signature excluding only CUDA_VISIBLE_DEVICES. All software,
hardware name/capability/memory, logical device index and math flags still match
strictly before restoration. Model/Adam/RNG/history restoration is unchanged;
every actual physical GPU/UUID/worker attempt is recorded in an append-only
ledger copied into the final artifact. Require homogeneous server GPU hardware
in CPU admission. Cross-physical-GPU bitwise trajectory equivalence is not claimed.
No existing checkpoint or common SW0148 implementation is modified.

Freeze all three fixed320/native32 predictions and both endpoints before new GT
scoring. Primary gate is standalone FG mean above SW0186 with positive paired-image
95%CI lower bound, plus all three means above comparable Slot70k. Report SW0188,
both IoUs, all seeds, conditional output and FG.87 separately. Independently
recompute actual masks/GT/means/intervals and verify trained CNN/Adam artifacts.
Raw imagewise scores remain private; publish code, cohort summaries and audit.
If the gate fails, close this recipe without a small margin/rate/epoch sweep.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `f244f4a83f79679b5d9f0c9fc6945ac3f5a16335c3079be24e5dd7d7721c66af`

````markdown
# SW0191 — production-partition error feedback

Completed and independently audited. Standalone three-seed means:
FG-ARI .799773 / foreground IoU .478037 / object IoU .659912.
All three decreased versus SW0186; this recipe is closed without a small sweep.
See RESULT_KO.md and results_archive/score_v1/summary.json. SW0186 remains best.
Keep the SW0186 incumbent and production masks, adding a class-balanced loss
on actual split/merged TRAIN patch pairs. Mined labels stay fixed across each
proposal's gradient and acceptance measurements. Separately check that this
auxiliary loss reaches emitted spikes, beyond the existing QCE gradient.

See [registered design](DESIGN_KO.md). Native32 joint70k, FG.87, automatic count
prediction and gate-causality proof remain outstanding. No new performance claim.

Seed0 training completed: all64 registered proposals were measured, with49
accepted updates. The completed model, Adam steps, frozen feedback contexts and
actual GPU ledger passed artifact validation; text receipts are in
results_archive/completed_seed0_v1. Seed2 also completed all64 proposals, with38
accepted updates and the same completed-artifact checks. Seed0 fixed320 masks
are frozen and validated without GT; receipts are in results_archive/frozen_seed0_v1.
All three fixed320 predictions and the independent score audit are now complete.

Seed1 completed all64 proposals with50 accepted updates. Its actual worker ledger
records GPU1, GPU1, then GPU2; the final history preserves the original first-two
model/Adam/RNG transaction digests. This exercised continuation on another
physical GPU without a fresh experiment restart. It does not establish bitwise
equivalence to uninterrupted training. All-seed receipts and hashes are in
results_archive/all_training_v1/completion_summary.json.

Local/server47 tests and six actual CLIs passed. All three exact incumbent CPU initializations and four-GPU hardware compatibility were admitted before launch. Eight actual supervisors were verified live. All three seeds' first-two real proposals passed frozen-context full-forward parity, emitted-S and feedback-only emitted-S/trainable-parameter credit, five-family total credit and committed CNN/Adam learning. Scoped receipts are in results_archive/cpu_v1 and start_v1. No new FG result or gate-causality proof is claimed.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `f9a78200eb3234a05f9618a377317cd877d5eebe7640a42bf5e9de67d9a64840`

````markdown
# SW0191 완료: partition-error feedback recipe 미채택

고정320장, native32 직접 예측, seed0/1/2 평균, standalone 기준:

| 모델 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW0191 QCE + actual-partition feedback | .799773 | .478037 | .659912 |
| SW0186 기존 최고 | .834681 | .569040 | .689802 |
| SW0188 QCE 추가 학습 | .831401 | .559170 | .681908 |
| comparable Slot70k | .781652 | .204360 | .203165 |

SW0191 seed별 점수:

| seed | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 0 | .796667 | .504761 | .652858 |
| 1 | .815104 | .498972 | .688736 |
| 2 | .787547 | .430379 | .638141 |

SW0186 대비 차이와 paired-image bootstrap95%CI (세 seed를 이미지별 평균 후 재표집):

- FG-ARI: -.034908, CI[-.041982,-.027848]
- Foreground IoU: -.091002, CI[-.104698,-.077117]
- Object IoU: -.029891, CI[-.036441,-.023436]

Conditional endpoint 평균은 .778284 / .609975 / .643924다. 새 loss가 실제
emitted spike와 trainable parameter에 gradient를 전달하고 CNN/Adam을 갱신한
것은 세 seed에서 검증했다. 그러나 이 구현과 weight1 추가 학습은 기존 최고보다
세 지표를 악화시켰다. Gradient 연결 증명이 일반화 성능이나 gating 인과적 기여를
증명하지는 않는다. 원인을 이 결과만으로 특정하지 않는다.

모든 seed는 같은 seed의 완료된 SW0186 전체 weights에서 시작했고 Adam은
다섯 family 모두 초기화했다. TRAIN512를 두 번 노출한64 proposals 중49/50/38을
엄격한 같은-RNG full-forward loss 감소 기준으로 수용했다. 추가 TRAIN512 예산은
기존70k foundation 학습 이력과 별개인 supervised native32 adaptation이다.
새 native32 joint70k 학습이나 unsupervised 성과로 해석하지 않는다.

세 seed 두 endpoint 예측을 전부 동결한 후 GT를 열었다. 독립 검증은 실제
후보1920 endpoint/image rows와 Slot960 rows를 다시 계산하고 means/CI/gates 및
학습 manifest를 확인했다. 원본 이미지별 점수는 비공개로 보관하고 집계와 검증만
공유한다. 정확한 검증 오차와 해시는 score_v1/completion_audit.json을 따른다.

최종 score SHA256:
`dec5e299a64f09949a32e40f3f0909d2f432afde0f30ffd71392871360c2a8b7`

독립 audit SHA256:
`4d8276c56a8cd174f746981e2d69f4bf4b40166e79b12ccfa9648e5ac349def6`

이 정확한 recipe는 종료하고 작은 weight/LR/margin 재탐색을 하지 않는다.
현재 최고 모델은 SW0186으로 유지한다. 사용자의 새64x64 FG-ARI>=.90 방향에
따라 SW0192는 SW0186 encoder/graph/gates/QCE를 유지한 native64 추가 학습의
메모리·시간·gradient 검증부터 진행한다.

````

</details>

## results_archive/cpu_v1/registered_README.md

<details>
<summary>registered_README.md</summary>

Source SHA256: `4f9874cf07ab1bb71a18e15d42c590bc10c7ac6dbf337c211e36c2653f796afe`

````markdown
# SW0191 — production-partition error feedback

Implementation/preflight in progress; no GPU training started.
Keep the SW0186 incumbent and production masks, adding a class-balanced loss
on actual split/merged TRAIN patch pairs. Mined labels stay fixed across each
proposal's gradient and acceptance measurements. Separately check that this
auxiliary loss reaches emitted spikes, beyond the existing QCE gradient.

See [registered design](DESIGN_KO.md). Native32 joint70k, FG.87, automatic count
prediction and gate-causality proof remain outstanding. No new performance claim.

````

</details>
