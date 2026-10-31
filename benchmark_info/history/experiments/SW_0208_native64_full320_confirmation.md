# SW_0208_native64_full320_confirmation

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `851292437f13263ce899299375e5f6dcd9cd15f9c994d99846d8fa1031c54c00`

````markdown
# SW0208: 204와 warm 모델의 전체320 확대 평가

204의 seed0 고정16 FG-ARI .900539와 동일16 warm192 .924401을
전체320으로 확대한다. 추가 학습은 하지 않는다. 두 모델은 독립 seed가
아니며, 본 실험만으로 세 seed 평균 .90 또는 70,000장 신규 학습 성공을
주장하지 않는다. supervised 모델이며 unsupervised 기록에 넣지 않는다.

- 입력: canonical HDF5 전체128 RGB, IDs1320..1639, crop 없음.
- 출력: native64 4096 patch, 기존1024-step spike readout/settle512 그대로.
- 모델: 실제204 완료 checkpoint와 실제192 TRAIN8 완료 checkpoint.
- RNG: 두 arm 모두 이미지 offset별193000+offset. 기존193/204 mini와
  같은 구간은 같은 RNG이며, 나머지288장의 RNG는 이번에 미리 확장 등록한다.
- 분할: offset modulo4의 네 작업. 각 작업은 두 arm을 실행한다. 모든
  GPU0..3에 exclusive lease로 대기하며 외부 사용자가 들어오면 양보한다.
- GT: 예측 전체640개 검증 후 mask를 읽는다. 2px modal, 동률 작은ID.
  기존 comparable Slot 세 seed의 전체320 native128 예측을 modal64로 평가.
- 모델/입력/소스/각 행 hash를 기록한다. 기존 mini와 겹치는 모든 행의
  label map은 정확히 같아야 한다. 점수는 독립 구현과 일치해야 한다.
- 점수 세 개와 paired bootstrap95% CI(10000회, seed208)를 공개한다.
  확대 학습 우선 후보는 전체320 FG-ARI가 높은 arm이다. 선택에 쓴
  validation320 결과는 독립 test 성능이라고 부르지 않는다.
- 70,000장 확대 학습은 이 평가의 후속 단계이며 현재 실행 내용과
  혼동하지 않는다. 전체320 seed0 .90도 세 seed 목표 달성은 아니다.

이 실험은 mini 개선 gate를 우회해 개선을 주장하는 실험이 아니다.
사용자가 요청한 절대 .90 결과의 범위를 확인하는 별도 confirmation이다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `8fb19c3737f8b68461ac3239a71489539889322eb645545b2d9240f7390d6705`

````markdown
# SW0208: full320 confirmation

**COMPLETE.** Native64 fixed320 seed0: warm192
**.877709/.791224/.696284**; component204 **.876242/.790093/.693262**.
Metric order: FG-ARI/foregroundIoU/matched-objectIoU. Neither reaches .90 on
all320. The three paired204-minus-warm95%CIs include zero. Keep warm192 as the
slightly higher-mean seed0 reference; no statistically established recipe gain.
See [RESULT_KO.md](RESULT_KO.md) and
`results_archive/completed_v1/score_v1/summary.json`.

Comparable Slot three-seed mean: .786902/.198201/.195203; Slot best FG seed2
.878278. Our two arms are seed0, supervised and inherited70k lineages, with
only8/16 additional native64 TRAIN images. No new70k training, independent
three-seed native64 win, unsupervised improvement or .90 goal is established.
All640 masks and all48 overlapping mini masks were checked; production and
independent metric implementations agree within2.22e-16. All queues ended.

## Historical registration

Registered on Frontier after11 local and11 Linux tests, AST/CLI checks and
actual checkpoint/full320 canonical RGB CPU admission. See DESIGN_KO.md.
This experiment compares actual204 and stronger warm192 on all320 images,
without further training. Seed0 only; it cannot establish a three-seed .90 claim.

Four exclusive-GPU shard supervisors2115329/2115330/2115331/2115332
were verified alive at registration. Each owns80 fixed images and evaluates
both models. All GPUs0..3 are eligible; foreign processes retain priority.
Completed rows are immutable and resume after an ownership interruption.
The last completed shard triggers GT-last scoring and matched full320 Slot
comparison. Registration artifacts: results_archive/registration_receipt.json
and results_archive/registration_v1/admission.json. No new70k training has begun.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `d14e0be71b70dcadbe7a211d1b77a354a5dfbcfdb4bbe85fdee14bc0eb0d218a`

````markdown
# SW0208: 고정320장 확대 평가 완료

64×64 patch, 전체128 RGB/no crop, validation IDs1320–1639,
2px modal GT(동률 작은ID), 실제1024-step spike mask/settle512다.
두 모델의 모든320장 예측과 Slot 세 seed 예측을 검증한 뒤 평가 GT를 읽었다.
새 학습이나 threshold 변경은 없다. 등록된 원본193/204 mini와 겹치는
48개 mask는 정확히 재현됐다. 독립 metric 구현과 최대 오차2.22e-16이다.

| 모델 | seed 범위 | FG-ARI | Foreground IoU | Object IoU |
|---|---|---:|---:|---:|
| 기준 warm192 | 0 | **0.877709** | **0.791224** | **0.696284** |
| component-credit204 | 0 | 0.876242 | 0.790093 | 0.693262 |
| comparable Slot Attention | 0 | 0.636265 | 0.205508 | 0.184986 |
| comparable Slot Attention | 1 | 0.846161 | 0.197193 | 0.212411 |
| comparable Slot Attention | 2 | 0.878278 | 0.191901 | 0.188211 |
| comparable Slot Attention | 0/1/2 평균 | 0.786902 | 0.198201 | 0.195203 |

204 minus warm의 image-paired bootstrap95% CI(10000회, seed208):

- FG-ARI: −0.001467, CI[−0.006263,+0.003277].
- Foreground IoU: −0.001131, CI[−0.004649,+0.002560].
- Object IoU: −0.003022, CI[−0.008057,+0.001870].

모든 CI가0을 포함한다. warm의 세 평균이 조금 높으므로 등록한 선택 규칙에
따라 후속 학습의 seed0 기준 checkpoint는 warm192로 유지한다. 이 비교는
학습량과 objective가 다른 두 checkpoint의 확인이며, 특정 loss가 성능을
해친다는 독립적인 인과 실험이 아니다. 동일 학습량의202 control은 기존
204 mini 실험에 있으며 이번 전체320 비교에는 포함하지 않았다.

## 16장의 .90은 무엇을 뜻했나

같은 고정16장에서는204 .900539, warm .924401이었다. 전체320장에서는
각각 .876242/.877709다. **16장에서 .90을 넘은 것은 사실이지만 전체320
성능이 .90인 것은 아니다.** 평가 이미지 수를 늘린 결과이므로, 학습
데이터 증가가 성능을 해쳤다거나 train overfit이 입증됐다고 해석하면 안 된다.

우리 seed0는 Slot 세 seed 평균보다 높지만, Slot의 최고 FG seed2 .878278과
비슷하다. 우리 독립 seed1/2 전체320 결과가 없으므로 세 seed 평균 승리나
native64 .90 목표 달성을 주장하지 않는다. 이320장은 반복 선택에 사용한
validation이며 untouched test도 아니다.

## 학습 범위와 다음 기준점

두 모델은 기존70k core 학습 lineage와32-grid encoder/core 학습을 물려받은
**supervised** 모델이다. 그 뒤 native64 적응에 사용한 추가 고유 TRAIN
이미지는 warm8장/20416장이다. 이번208은0 update의 평가만 수행했다.
새로운 native64 70,000장 학습 완료 결과로 표기하지 않는다.

후속 기준은 실제 warm checkpoint의 .877709/.791224/.696284다. 새70k
학습은 아직 등록하지 않았다. 기존204 기록의 B4 proposal은 약441–445초,
gradient forward/backward만 약285–287초였다. 이 microbatch1 절차를
70k 전체에 단순 반복하면 약90 GPU-days/한 pass의 계산량이므로,
현재 네 GPU 큐가 단일 학습을 이미 네 GPU로 병렬화한다고 해석해서는 안 된다.
이 수치는 소수 batch에서 외삽한 비용이며 실제70k 소요 시간 측정이 아니다.
따라서 확대 학습은 geometry/최적화/비용을 명시한 별도 실행으로 기록해야 한다.

증빙: `results_archive/completed_v1/`의 summary, 여덟 frozen shard protocol,
네 queue 상태·GPU attempt ledger, archive receipt. 데이터·checkpoint·실제
mask 배열과 imagewise 점수는 GitHub에 포함하지 않았다. Goal은 active로
유지하며, gating 인과 기여·데이터 scaling·세 seed .90은 계속 미검증이다.

````

</details>

## SUPPORT_DIAGNOSIS_KO.md

<details>
<summary>SUPPORT_DIAGNOSIS_KO.md</summary>

Source SHA256: `50556889a67c19079c32b7d065c5d94d0e5e28299570f1b5a4d0efe2e5181e0c`

````markdown
# 완료된 mask의 posthoc 진단: grouping을 바꿀 여지가 있는가

**아래 GT-assisted reference는 실제 모델 예측이나 달성한 성능이 아니다.**
기존 예측의 전경/배경 membership을 고정하고, 전경 patch를 GT 물체별로
묶는 구체적인 진단용 partition을 만들었다. 특정 classifier가 이를 학습할
수 있다는 보장이나 새 .90 이상 모델 결과로 해석하지 않는다.

| 모델과 평가 범위 | 실제 FG-ARI | GT로 전경 묶기만 수정한 reference | 놓친 GT 전경 patch 비율 |
|---|---:|---:|---:|
| warm192 seed0 전체320 | 0.877709 | 0.956939 | 6.37% |
| warm192 seed0 공통16 | 0.924401 | 0.959205 | 6.16% |
| warm205 seed1 공통16 | 0.815856 | 0.936003 | 8.69% |
| warm205 seed2 공통16 | 0.788414 | 0.976234 | 3.17% |

특히 seed2 공통16에서 전경 안의 same-object split rate는.091640,
different-object merge rate는.118190이다. 전경 누락만 GT로 보완하는
별도 attachment reference는.799047로, 현재 .788414보다 조금 높다.
반면 membership을 고정한 GT-pure grouping reference는.976234다.
이는 classifier의 split/merge를 다루는 controlled test를 진행할 근거이며,
GT가 없는 실제 예측에서 이 개선이 가능하다는 증거는 아니다.

두 reference는 서로 다른 GT-assisted 조작이며 효과를 더해 합산하지 않는다.
이 보고서는 위의 구체적 reference를 계산했을 뿐 새 전역 optimum/ceiling
정리를 증명하지 않는다. 모든 source masks/GT modal/metric은 독립 검증했고,
전경 고정 reference의 foreground IoU가 원본과 정확히 같음을 검사했다.
추가 model forward/optimizer update는 모두0이다. 실행 중인209/210의
source, model, readout과 사전 판정은 바꾸지 않았다.

원본: `results_archive/support_diagnosis_v1/summary.json`.
재현: `python -m collaborative_test.SW_0208_native64_full320_confirmation.diagnose_fixed_support`.

````

</details>
