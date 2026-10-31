# SW_0249_four_model_35000

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `2c30cdfce278e074fc04d9d2252f8b2bdf6f2b03c89a4f2facebfd6be6bb45fc`

````markdown
# SW0249: four frozen models, matched additional TRAIN35,000

The user ends open-ended model/loss improvement. Compare SW0240, SW0237,
SW0244-within16 and SW0245-within16 only. Twelve model/seed jobs share the
same deterministic35,000 TRAIN IDs, ten complete shuffled passes, Adam reset,
learning rates3e-5/3e-4/3e-4, clipping1, native64, rollout1024/settle512 and
fixed320 evaluation IDs1320..1639. Model-specific existing loss coefficients
remain1/1/16/16. No loss or architecture sweep and no candidate-screen gate.

The new adaptation stage uses the same trainable whole encoder and sensory
projection for all models; all other core states and the readout remain frozen.
This standardizes new training conditions, rather than reproducing the
encoder-only freeze policy of244/245's historical128-update stages.
All use the unchanged original217 teacher and established mask supervision.
SW0240 has already received70k additional training; this inherited history is
retained and reported, so the comparison does not equalize lifetime compute.
SW0244 and245 share a recipe and are not independent architectural hypotheses.

35,000 is not divisible by16: use2,187 full batches and one8-image batch per
pass, ten passes,21,880 updates and350,000 presentations exactly. Never pad,
drop images, or silently substitute35,008/34,992. Shuffle matches across models
within a seed. The35k subset is sampled once from the original70k TRAIN pool;
reserved IDs1000..1639 are excluded.

User explicitly authorized reproducing SW0244's missing seed0/2 using its
original128-update encoder-only within16 recipe, original64 TRAIN IDs/teacher,
and original244 RNG seed convention. These are labeled new reproductions,
not original historical checkpoints. Reproduction updates are reported
separately from the common35k stage. Seed1 uses the actual244 checkpoint.

Each job completes CPU source/gradient and GPU loss/readout admission before
training. Independent worker checkpoints contain encoder, actual projection,
Adam, RNG, history and immutable bindings; a failed job retains its logs and
last committed boundary. No foreign process arrival triggers yielding. Shared
GPU admission requires remaining memory for the unchanged80%-allocation cap
plus512MiB headroom. Cooperating jobs still lease one worker per physical GPU.

Retain intermediate full-state checkpoints near first-pass2,500 and10,000
distinct images and at complete passes1/2/4/6/8/10. They permit later learning
curve evaluation without retraining. Do not infer score saturation from loss
alone; this fixed final comparison does not select checkpoints using scores.

All twelve final prediction protocols and Slot references precede GT scoring.
Compute all three metrics, paired image contrasts and per-seed values. Audit
actual final Adam counters/history, checkpoint state and raw masks. Archive
aggregate results and share research text only. End the goal after all twelve
jobs, scoring, audit and final report; do not queue exploratory followups.

````

</details>

## RESOURCE_POLICY_V3.md

<details>
<summary>RESOURCE_POLICY_V3.md</summary>

Source SHA256: `bfaf6dddc6e3d310d4e3533e7040becdd5ccd38177c182959005e6943e880d2f`

````markdown
# Shared GPU admission based on the running worker's observed footprint

The initial admission rule reserved the worker's full80% allocation cap plus
512MiB, approximately20GiB on24GiB hardware. This excluded GPU3 despite its
approximately17GiB free memory. Two workers had already passed actual native64
spike preflight and training, with observed resident memory2,234MiB each.

The resource-only v3 wrapper requires at least8GiB free before admission,
exceeding three times that observed residency plus headroom. This threshold is
not a claimed upper bound on allocation. Each unchanged worker still runs its
actual GPU gradient/spike preflight, retains its original80% PyTorch allocation
cap, and preserves the existing OOM retry/checkpoint policy. No foreign-process
arrival causes yielding; no other user's process is signalled.

Only ten waiting supervisors were replaced. Their exact identities and start
times were checked. Their old state/admission logs were archived, and replacement
queues resumed the same job paths under the same GPU/queue leases. The two active
training workers were untouched. The owned coordinator was briefly suspended
only to make this waiting-queue handoff atomic to its observer, then resumed.

The nine scientific-source files and their registered fingerprint were verified
unchanged before/after handoff. Batch, data, loss, optimizer, runtime math flags,
architecture, checkpoint state and readout are unchanged. The resource wrapper
and tests have their own SHA manifest retained privately. Both local and Linux
resource tests passed. This document supersedes the initial admission paragraph
of DESIGN_KO.md without editing that immutable scientific-source fingerprint.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `2c1507131955c055a93ecd16d0c21edf216f70a5cc33daad3a1e2f5e454e7eda`

````markdown
# SW0249 최종 네 모델 비교

동일한 TRAIN 35,000장으로 각 모델을 10회 추가 학습하고, 고정 320장·native64·seed 0/1/2에서 평가했다. 총 12개 학습·평가와 독립 검증을 완료했다.

| 모델 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW240 | 0.889187 | 0.754775 | 0.686438 |
| SW237 | 0.887640 | 0.718838 | 0.682593 |
| SW244 | 0.879873 | 0.788644 | 0.666593 |
| SW245 | 0.879739 | 0.788727 | 0.666623 |
| 기존 Slot Attention 참고값 | 0.786902 | 0.198201 | 0.195203 |

FG-ARI와 Object IoU의 평균은 240번이 가장 높았다. Foreground IoU는 245번이 가장 높고 244번과 거의 같았다. 237번과 240번의 FG-ARI 평균 차이는 -0.001547이며, 이미지 단위 paired bootstrap 95% 구간은 [-0.004595, +0.001479]이다. 이 구간은 0을 포함하므로 두 모델의 차이를 확실한 우열로 단정하지 않는다. 이 구간은 seed 모집단의 불확실성을 추정한 것이 아니다.

## Seed별 결과

| 모델 | Seed | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|---:|
| 240 | 0 | 0.891040 | 0.833238 | 0.701442 |
| 240 | 1 | 0.897328 | 0.753545 | 0.692287 |
| 240 | 2 | 0.879194 | 0.677543 | 0.665585 |
| 237 | 0 | 0.893587 | 0.820960 | 0.698422 |
| 237 | 1 | 0.894242 | 0.729373 | 0.686182 |
| 237 | 2 | 0.875091 | 0.606180 | 0.663174 |
| 244 | 0 | 0.888876 | 0.814499 | 0.680433 |
| 244 | 1 | 0.886121 | 0.754822 | 0.670973 |
| 244 | 2 | 0.864621 | 0.796610 | 0.648374 |
| 245 | 0 | 0.888421 | 0.814443 | 0.679792 |
| 245 | 1 | 0.886177 | 0.755131 | 0.671701 |
| 245 | 2 | 0.864621 | 0.796607 | 0.648377 |
| slot | 0 | 0.636265 | 0.205508 | 0.184986 |
| slot | 1 | 0.846161 | 0.197193 | 0.212411 |
| slot | 2 | 0.878278 | 0.191901 | 0.188211 |

## 조건과 검증

- 동일한 35,000개 TRAIN ID, 10회 전체 순회, 350,000장 노출, 21,880회 업데이트. Batch 16이며 마지막 batch 8장도 포함했다.
- 네 모델 모두 fresh Adam과 동일 LR(3e-5 / 3e-4 / 3e-4), gradient clip 1.0을 사용했다. Encoder 전체와 실제 sensory projection을 학습하고, 나머지 core/readout은 고정했다.
- 구조·readout·기존 loss 유지: 237/240 base 1, 244/245 within 16. 사용자 승인대로 244의 누락 seed 0/2를 원래 128-update 설정으로 재현했다.
- 중간 validation 점수로 checkpoint를 선택하지 않고, 미리 정한 10회 종료 checkpoint를 평가했다. 12개 최종 checkpoint를 CPU에 실제 복원하고 Adam step·optimizer group·학습 ID 이력·고정 core/teacher·모델 SHA를 확인했다.
- 우리 모델 3,840개와 기존 Slot 960개, 총 4,800개 mask의 세 지표를 독립 재계산했다. 최대 오차는 3.33e-16였다.
- 모든 예측 protocol이 평가 정답을 읽기 전에 확정되었다. 평가 ID는 1320–1639이며 학습 ID와 겹치지 않는다.

## 해석 범위

이번 추가 학습 예산은 같지만 이전 학습 이력은 다르다. 특히 240번은 이미 70,000장 추가 학습을 거쳤다. 네 모델 모두 mask supervision을 사용했다. Slot Attention은 기존 reconstruction 학습 checkpoint의 참고 결과이며 이번 35,000장 추가 학습 비교에 참여하지 않았다. 따라서 이 표는 unsupervised 우위나 동일 총학습량의 구조 비교를 증명하지 않는다.

고정 320장은 반복 사용한 validation이며 새로운 untouched test가 아니다. 각 gate의 인과적 기여나 데이터 수 증가의 효과도 이 최종 비교만으로 증명하지 않는다.

## 보관과 종료

정확한 조건·seed별 점수·paired bootstrap·checkpoint 검증은 `PUBLIC_RESULT.json`에 있다. 로컬 checkpoint 참조는 `collaborative_testing/collaborative_private_runtime_20261008/SW0249_final_registry/reference.json`에, 미리 정한 seed0 평가 이미지 1320/1321/1322의 비교 그림과 원본 배열은 `collaborative_testing/generated/visualizations/SW0249_four_model_final/`에 보관했다. 원본 데이터·checkpoint·개인 실행 경로는 GitHub 공개 대상에 포함하지 않는다.

서버의 12개 작업과 coordinator는 정상 종료했다. 다른 사용자 프로세스는 변경하지 않았다. 후속 모델·loss 탐색이나 추가 학습은 등록하지 않았다.

````

</details>
