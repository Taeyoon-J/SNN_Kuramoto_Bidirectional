# SW_0181_actual_descent_qce

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `001d7b723ffbcaef5a1d2fc30f0f196a6425027cd0a58efe911388331cb208a6`

````markdown
# SW0181 — 실제 QCE loss를 낮춘 Adam proposal만 반영

SW0180 동일 TRAIN8 replay에서 seed 0·1은 QCE와 standalone FG가 개선됐지만
seed 2는 loss 0.526248→0.936206, FG 0.784225→0.636125로 악화됐습니다.
이번 가설은 실제 학습 목적을 악화시키는 optimizer proposal을 반영하지 않으면
세 seed의 학습 안정성과 최종 mask 성능이 개선되는지 확인하는 것입니다.
현재 minibatch descent가 다른 이미지의 loss나 FG 개선을 보장하지는 않습니다.

## 변경 범위와 고정 예산

세 seed 0/1/2 모두 자기 fresh SW0097 SourceT+zero-context에서 시작합니다.
공통 SW0171 TRAIN512 RGB/full modal GT, shared RandomState(171001)의 두
연속 permutation, logical B16/micro4, T1024/settle512/live-tail64,
product-Q LOO class-balanced CE(temp 0.1), 기존 네 optimizer family와 LR,
global clip 1 및 기존 parameter projection을 유지합니다. Factory/context/update
seed law는 SW0179와 동일합니다. Seed 1 초기 tensor는 SW0178의 f372…와 일치해야 합니다.
Core/gate/encoder 구조, mask readout, 배경 정책을 바꾸지 않습니다.

각 seed는 **64 proposals**만 처리합니다. 거절됐다고 추가 batch·LR·scale을
시도하지 않고 다음 등록 batch로 갑니다. 64 accepted updates가 될 때까지
늘리지 않습니다. Accepted/skipped, optimizer 실제 step, proposal 이미지 노출,
추가 no-grad 검사 연산량을 분리해 기록합니다. 이 실험은 supervised이며
native32/70,000장 신규 학습이나 unsupervised 성능으로 해석하지 않습니다.

## 반영 또는 복원

각 proposal에서 예정 RNG를 설정하고 gradient를 비운 뒤 모델 parameters/
buffers, Adam moments/step/groups와 CPU·CUDA·Python·NumPy RNG를 깊은 복사합니다.
기존 SW0179 gradient helper로 원래 Adam proposal을 정확히 한 번 만듭니다.
그 helper의 before loss는 gradient 선형 예측이 아닌 실제 전체 T1024 forward의
QCE 값입니다. Tail64는 backward 범위입니다. 처음 두 proposals에서는 동일
RNG의 별도 full no-grad before loss와 helper loss의 parity를 1e-6 절대+상대
오차 안에서 확인합니다. Parity 실패는 구현 검증 실패로 종료하며 loss를 바꾸지 않습니다.

매 proposal은 같은 batch와 시작 RNG로 full no-grad post loss를 계산합니다.
유한한 값이고 `post < before - max(1e-7, 1e-6*abs(before))`일 때만 반영합니다.
수치 guard는 사전 고정하며 sweep하지 않습니다. 반영하면 proposal의 모델·Adam을
유지하고 RNG를 gradient proposal 직후 상태로 되돌립니다. 거절하면 원래 모델·
Adam·시작 RNG를 완전히 복원합니다. 두 경우 모두 gradient를 비웁니다.
추가 no-grad 측정 자체가 모델 상태를 바꾸지 않았는지도 확인합니다.
비유한 post loss는 값을 null과 사유로 기록하고 복원하며 구현 예외를 숨기지 않습니다.

첫 두 proposals의 실제 S gradient와 네 family의 proposed gradient/delta를
검증합니다. Proposed 변경과 실제 committed 변경을 구분합니다. 거절된
proposal의 committed delta는 0이며, 이를 학습 성공이나 gate 기여로 해석하지 않습니다.
실제 Adam step은 accepted count와 같아야 합니다. Durable boundary는
proposal cursor와 accepted count를 모두 저장하며 rejected batch도 소비한 것으로
기록합니다. Resume은 둘을 섞거나 uncommitted gradient를 재사용하지 않습니다.

## 평가와 판정

완료된 SW0178/179 unconditional QCE의 같은 seed 초기 상태·데이터·loss·
batch/RNG 조건을 확인해 대조군으로 사용합니다. 새 후보의 모든 고정320
예측·protocol/hash를 먼저 확정한 뒤 평가 정답을 읽습니다. 실제 S→native Q→
production QCC→SW0165 standalone이 primary이며 conditional은 진단용입니다.
세 seed 평균 FG는 SW0165 0.8042989149474198와 matched Slot 0.7816519719609135를
엄격히 초과하고, 두 paired FG 95% CI lower가 0보다 커야 합니다. 두 IoU 평균도
Slot 0.20436012776655546 / 0.2031649861884501보다 엄격히 커야 합니다.
Bootstrap은 세 seed의 같은 이미지별 평균 후 320장을 10,000회 재표집합니다(seed 178).
FG 0.87 목표는 별도로 보고하며 실패를 숨기거나 사후 seed를 고르지 않습니다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `b8893c737c051bdd9fc2719551d95cd9c5856ab857a7616cd40fbb26a5ccea82`

````markdown
# SW0181 — actual-loss acceptance for QCE training

**Completed and independently audited:** standalone seed0/1/2 mean
**0.814836 / 0.589302 / 0.671681**. The registered primary gate passed; all three
metrics exceed the matched Slot baseline. This is mask-supervised TRAIN512
fine-tuning. See [the final result and limitations](RESULT_KO.md).

SW0180 found that seed 2 deteriorated on the same eight TRAIN images after
unconditional QCE updates: CE 0.526248 → 0.936206 and standalone FG-ARI
0.784225 → 0.636125. Seeds 0 and 1 improved on that replay. This small diagnostic
supports testing optimization stability; it does not establish a population-wide
cause or prove that minibatch descent will improve segmentation.

SW0181 retains the source, full forward, QCE objective, four optimizer families,
learning rates, shared TRAIN512 order, and 64-proposal budget. It commits an Adam
proposal only when the same batch and RNG produce a strictly lower actual loss.
Otherwise it restores model and Adam state and consumes the batch without a retry.
All three seeds receive the same policy. See [the preregistered design](DESIGN_KO.md).

## Verified evidence

- The four transaction unit tests passed on Frontier.
- On real seed 0/1/2 model copies, both empty and populated Adam states and model
  parameters/buffers restored exactly after synthetic perturbations. Original
  checkpoint files remained unchanged.
- This CPU proof used synthetic gradients only. It performed no spike forward,
  committed no training updates, initialized no CUDA, and read no validation data.
- Receipt: [CPU transaction proof](results_archive/cpu_v1/transaction_cpu_receipt.json).

The server's full 16-test suite and an actual subprocess CLI check passed.
An earlier undefined seed-choice constant stopped the first supervisor before
queue creation or training; the failure log and corrected-file hashes are
preserved in [the deployment receipt](results_archive/cpu_v2/training_cpu_receipt.json).

All three actual GPU workers started on GPUs 1/2/3. The first two proposals per
seed passed emitted-spike and four-family gradient checks. Same-RNG full no-grad
before loss differed from the gradient-forward value by at most 4.66e-9, below
the fixed acceptance guard. Both acceptance and rejection occurred, and actual
serialized Adam steps matched the accepted count independently of the proposal
cursor. This proves that the policy is executing, not that segmentation improved.

All three training runs completed and their model/optimizer/history artifacts
passed validation. Each seed consumed the registered 64 proposals, with 32/38/34
accepted updates for seeds 0/1/2 respectively. The remaining proposals were
rejected without replacement. This is fewer committed Adam steps than the
unconditional control, with the same proposal budget and unique TRAIN512 images.
Each run additionally performed 64 post checks and two pre checks, for 130
logical B16 full forwards (2,080 image forwards) before any prediction.
See [training records](results_archive/train_v1/seed0_training_record.json).

Fixed320 prediction and evaluation completed for all three seeds. The final
independent audit recomputed both endpoints, per-image metrics, means, paired
bootstrap and gates, and verified immutable artifact hashes.
Three prediction supervisors and the GT-last score coordinator were queued and
all completed. Each prediction waited for its own completed training artifact,
and scoring waited for all three immutable predictions.
The deployed evaluation suite passed 32 tests and four actual subprocess CLI checks.
The supplemental unconditional-QCE comparator passed two additional tests and
validated the actual pinned SW0179 score archive without reading validation masks.
Its matched-seed comparison also completed after the SW0181 score was frozen.

See [evaluation deployment verification](results_archive/eval_v1/cpu_receipt.json)
and [queue launch record](results_archive/eval_v1/launch.json).

This is a mask-supervised experiment, not an unsupervised leaderboard update,
fresh native32/70k training, or evidence that individual gates are necessary.
SW0165 remains the unsupervised incumbent.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `20e8f2f0b174eea57e4a01690275b1ba72380ae1c9ac517a27d35dc4f60b31af`

````markdown
# SW0181 결과 — 실제 loss 감소를 확인한 업데이트만 반영

고정 native32(32×32 patch), 공통 평가 320장, seed 0/1/2 평균입니다.
실제 spike → native Q → production QCC → SW0165 standalone mask를 평가했습니다.
세 seed의 예측과 protocol을 모두 확정한 뒤 평가 정답을 읽었습니다.

| 설정 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW0181, mask-supervised | **0.814836** | **0.589302** | **0.671681** |
| SW0178/179, 동일 seed unconditional QCE 대조군 | 0.761620 | 0.469308 | 0.584656 |
| SW0165, 기존 unsupervised 기준점 | 0.804299 | 0.609975 | 0.651149 |
| Slot Attention, 우리 70k 학습 epoch10 기준 | 0.781652 | 0.204360 | 0.203165 |

SW0181은 Slot Attention을 세 지표 모두 초과했고 등록한 primary gate를
통과했습니다. SW0165보다 FG-ARI와 Object IoU는 높지만 Foreground IoU는 낮습니다.
정답 mask를 사용한 추가 학습 결과이므로 unsupervised 모델의 승리로 합치지 않습니다.

## Seed별 결과와 신뢰구간

| seed | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 0 | 0.848596 | 0.724047 | 0.699976 |
| 1 | 0.796650 | 0.592035 | 0.678430 |
| 2 | 0.799263 | 0.451825 | 0.636637 |

세 seed의 같은 이미지별 점수를 먼저 평균하고, 공통 320장을 10,000회
paired bootstrap했습니다(RandomState seed 178).

- FG-ARI 차이, SW0181 − SW0165: **+0.010537**, 95% CI **[+0.005757, +0.015223]**.
- FG-ARI 차이, SW0181 − Slot: **+0.033184**, 95% CI **[+0.014304, +0.051693]**.
- FG-ARI 차이, SW0181 − unconditional QCE: **+0.053216**, 95% CI **[+0.044362, +0.062337]**.

Unconditional QCE 대비 seed 0/1은 각각 0.849988→0.848596,
0.818029→0.796650으로 낮아졌고, seed 2는 0.616842→0.799263으로 회복했습니다.
따라서 모든 seed가 좋아졌다는 결과는 아닙니다. 다만 동일 정책으로 세 seed를
실행했을 때 평균과 하위 seed의 붕괴 문제가 개선됐습니다. 최종 FG 분할은
minibatch QCE 감소만으로 보장되지 않습니다.

SourceT foreground를 고정한 conditional 진단 평균은
**0.802259 / 0.609975 / 0.655217**입니다. 이 endpoint는 foreground membership을
고정하므로 standalone과 구분합니다.

## 변경 내용과 비용

기존 SW0179와 같은 fresh SourceT+zero-context, 공통 TRAIN512, 두 permutation,
B16/micro4, T1024/settle512/live-tail64, QCE temp0.1, 네 optimizer family와 LR를
사용했습니다. 변경은 같은 batch·RNG에서 실제 post loss가 사전 guard보다 더
감소한 Adam proposal만 반영한 것입니다. 거절 시 가중치·Adam·RNG를 복원했습니다.

64 proposals 중 반영 수는 seed별 **32 / 38 / 34**, 거절 수는 **32 / 26 / 30**입니다.
거절된 batch를 대체하거나 학습 예산을 늘리지 않았습니다. 각 seed는 gradient
forward 64회 외에 post 검사 64회와 pre 검사 2회(B16)를 추가했습니다.
따라서 총 130 logical full forward, 2,080 image forwards가 필요했습니다.
기존 대조군과 committed Adam step 수 및 연산 비용이 같다는 주장은 하지 않습니다.

## 검증과 범위

서버의 실제 모델·Adam 산출물 검증, 첫 두 proposals의 실제 spike와 네 family의
gradient, no-grad loss parity, model/optimizer rollback 및 checkpoint cursor를
확인했습니다. 최종 독립 검산은 세 seed·두 endpoint의 실제 예측에서 모든
per-image metric, seed 평균, paired CI, 판정을 다시 계산했습니다. 평가 전후
예측·protocol·모델·optimizer·score hash도 동일했습니다.

이 결과는 **TRAIN512 mask-supervised 추가 학습**입니다. Native32에서 처음부터
70,000장 전체를 학습한 결과가 아니며, FG-ARI 0.87, 데이터 증가에 따른 향상,
개별 gate의 필요성·학습 기여는 아직 증명하지 않았습니다. Unsupervised 기준점은
SW0165를 유지합니다.

원본 점수 SHA256:
`3f69a74a5729341b5fd11f2bd74f2d51ff245c0839f76053b8733af9523ddd54`.

- [고정 평가 원본](results_archive/score_v1/evaluation.json)
- [독립 검산 결과](results_archive/score_v1/independent_audit.json)
- [독립 검산 코드](results_archive/score_v1/independent_verifier.py)
- [동일 seed QCE 대조군 비교](results_archive/score_v1/unconditional_comparison.json)
- [사전 등록 설계](DESIGN_KO.md)

````

</details>
