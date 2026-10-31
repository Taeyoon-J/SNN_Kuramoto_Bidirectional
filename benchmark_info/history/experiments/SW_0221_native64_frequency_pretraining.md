# SW_0221_native64_frequency_pretraining

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `63fb10aabe319ed6d9ecf14539c6fffda9b71a4acc42577d7dfaa40a5300cadb`

````markdown
# SW0221: 실제 감각 주파수의 물체별 일관성을 학습

기존 최고 모델(seed0 warm192, seed1/2 warm205)을 그대로 시작점으로 쓴다.
전체 encoder(base와 context)를 학습하고 core, gate, graph 파라미터, gamma 통계,
기존 spike/membrane readout은 고정한다. 최종 mask는 원래 1024-step 경로만 사용한다.
별도 RGB mask decoder나 Slot Attention head를 추가하지 않는다.

학습 대상은 raw gamma가 아니라 실제 gamma_to_drive의 projection, 공간 표준화,
tanh를 모두 통과한 (omega + freq_gain * drive) / (abs(freq_gain) * pi)다.
실제 freq_gain이 0이 아니므로 원형 위상 거리만으로 같은 주파수라고 취급하지 않는다.
GT 물체 내부 분산과 4-neighbor 접촉 물체 평균 간 거리의 unit-margin 제곱 hinge를
동일 가중치로 최소화한다. 배경도 하나의 class다. 멀리 떨어진 동일 외형 물체를
억지로 분리하지 않는다. 171번의 graph projection CE와 다른 입력 물리량이며,
당시 고정했던 base encoder도 이번에는 학습한다. 성공을 가정하지 않는다.

세 seed 각각 TRAIN64/512 두 arm, 128 update, B16, 노출 2048회로 고정한다.
기존 TRAIN512 순서의 prefix를 쓴다. base LR=3e-5/context LR=3e-4, clip=1,
fresh Adam, 마지막 128 step만 평가한다. 이는 mask-supervised encoder pretraining이며
core 공동 학습이나 새 native64 70,000장 학습이 아니다. 이전 70k/core와 native32
TRAIN512 supervision을 상속한다. 64/512 비교는 추가 native64 학습 단계의 데이터
다양성 비교이며 최초 학습 전체의 scaling 또는 새로운 고유 데이터 증가 증명이 아니다.

DEV는 기존 16장 IDs1320,1322,...1350, RNG193000+offset, canonical128/no crop,
2px modal GT. 여섯 arm의 코드와 최종 mask protocol을 모두 고정한 후 DEV GT를 읽는다.
학습 loss 감소와 입력 주파수 개선만으로 최종 mask 향상이나 gating 인과성을 주장하지 않는다.
FG-ARI 평균 +.01, paired image bootstrap CI 하한>0, Object IoU 평균 비감소인
arm만 전체320 검증 후보로 삼는다. 전체320 검증 전 기존 최고 기록과 .90 목표는 유지한다.
CPU admission, 실제 encoder checkpoint/resume, full scoring branch 검증 후 배포한다.

````

</details>

## PEER_REVIEW_KO.md

<details>
<summary>PEER_REVIEW_KO.md</summary>

Source SHA256: `bf41b3e39f7957069cebeede5590b1fd18ea5d6b3b969205736f955eeaf704f0`

````markdown
# 동료 연구 확인 — 2026-10-18

이번 사이클에서 origin/patch_v2를 다시 fetch했다. HEAD는
7a21632b93805f4f1da69a2e77a9ddd95c624d0a로 이전 검토와 같았다.
PV2_0102의 seed 간 collapse 및 101번 초기 headline 철회 관련 기존 검토를 유지한다.
한 seed 또는 짧은 학습만 보고 데이터 scaling 성공을 주장하지 않는다.
이번 시험도 세 독립 source seed와 동일 exposure의 두 데이터 arm을 모두 실행한다.
동료의 제안 단계인 graph freeze를 검증 완료 성과로 취급하거나 그대로 포팅하지 않는다.
이번에는 기존 graph dynamics와 gate 및 spike readout을 유지하고 입력 encoder만 바꾼다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `be9d92cee7f77af4f49fedee09a292745e6a92c863e82266f3a5d7fd7c37a416`

````markdown
# SW0221

Prospective six-arm mask-supervised encoder-frequency pretraining experiment.
See DESIGN_KO.md for the fixed protocol. No result is claimed before execution.

Actual execution2026-10-18:12 local/12 Linux tests,3 local CLI checks and all6 real source/TRAIN CPU admissions passed. Six exclusive idle GPU queues registered. START_EVIDENCE.json records actual NVIDIA-bound workers. All six final masks freeze before DEV GT scoring. No new performance claim; native64 full320 incumbent remains .848615/.592533/.672047.

A separate read-only diagnose_frequency.py waits for all six scored arms, revalidates their actual models and masks, then compares within-object frequency variance, adjacent-class separation and global variance with final spike mask scores. Independent centered NumPy statistics are checked against the actual Torch training objective. It adds no model forward or training and leaves the original six-arm fingerprints unchanged. Coherent and collapsed frequency fixtures passed locally and on Linux.

Completed: fixed16 three-seed baseline .842891/.600692/.702161; TRAIN64 .705741/.243287/.581092; TRAIN512 .693004/.239999/.574074. Both expansion gates fail. TRAIN and DEV within-object frequency variance increased despite lower total loss. The hard unit-margin branch is closed. Actual six encoder checkpoints and all metric means/bootstrap CIs audited; full320 incumbent .848615/.592533/.672047 unchanged. See RESULT_KO.md and completed_v1 summaries.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `99372fde49a23072a152d11a0729b65e3f2e8f50764fcb62f8cf865a007adc07`

````markdown
# SW0221: 분리 margin을 강제한 encoder 학습은 성능 하락

기존 최고 native64 세 source 모델에 같은 128 update/B16, fresh encoder Adam을
적용했다. core/gates/readout/feature 통계는 고정했다. 여섯 최종 mask protocol을
모두 동결한 뒤 기존 DEV16(1320,1322,...1350)의 2px modal GT를 읽었다.

| 추가 TRAIN 단계 | 세 seed 평균 FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 source | .842891 | .600692 | .702161 |
| TRAIN64 | .705741 | .243287 | .581092 |
| TRAIN512 | .693004 | .239999 | .574074 |

세 seed 각각에서도 두 arm 모두 세 지표가 내려갔다. 기준 대비 FG 차이는
TRAIN64 -.137150, paired image 95% CI[-.222561,-.065948],
TRAIN512 -.149887, CI[-.229969,-.079548]이다. 따라서 둘 다 확대 gate 실패다.
512-64 차이 -.012737의 CI[-.034828,.010643]는 0을 포함하므로,
이 결과를 일반적인 데이터 증가의 역효과라고 해석하지 않는다.

중요한 구별: TRAIN total loss는 감소했지만, 물체 내부 주파수 분산은 감소하지 않았다.
TRAIN64의 seed0/1/2 내부 분산은 .004660/.013782/.005229에서
.040941/.074095/.047621로 증가했다. TRAIN512도
.004777/.014256/.005424에서 .041751/.081332/.048474로 증가했다.
기존 입력의 내부 분산은 이미 작았고, 총 loss 감소는 주로 이웃 class 사이의
unit-distance margin 위반을 줄이는 항에서 나왔다. 총 loss 감소를
물체 내부 rhythm 일관성 개선이라고 주장할 수 없다.

해당 unit-margin encoder branch는 종료한다. LR/epoch/margin 수치 sweep이나
이 설정의 70,000장 확대는 하지 않는다. 다음 후보는 기존 source의 실제 물체별
주파수 중심을 보존하면서 내부 변동만 줄이고 원래 gamma 입력을 anchor로 유지하는
학습이다. 이는 검증 전 가설이며, 아직 새 모델 성능으로 취급하지 않는다.

이 시험은 mask-supervised 추가 native64 encoder 단계다. 상속한 70k/core 및
native32 TRAIN512 supervision이 있으므로 신규 고유 데이터 64/512 전체 학습이나
unsupervised scaling 증명이 아니다. native64 전체320 최고 세 seed 평균
.848615/.592533/.672047은 유지되며 .90 목표는 아직 미달이다.

후속 읽기 전용 frequency 진단도 완료했다. DEV의 물체 내부 분산은 TRAIN64의
seed0/1/2에서 .005064/.017437/.007019 -> .054533/.078117/.048307,
TRAIN512에서는 .052330/.088710/.050701로 증가했다. 두 arm 모두 DEV의
물체간 margin 항도 줄었으므로 단순히 TRAIN objective만 암기한 결과는 아니다.
기존에 작은 주파수 대역을 큰 폭으로 넓히면서, 물체 내부 변동도 함께 커졌다.
이 변화와 mask 하락은 확인했지만 개별 gate의 붕괴나 유일한 인과 원인을
이 진단만으로 특정하지 않는다.

실제 여섯 encoder checkpoint의 SHA/cursor/Adam128/history를 대조했고, 독립
post-score 진단에서 원래 source+최종 encoder 모델 여섯 개와 mask를 재검증했다.
144개 mask 평가 벡터의 평균/paired bootstrap CI 재계산 차이는 0,
두 metric 구현 최대 오차는 2.22e-16이다. 새 model forward/추가 학습은 0이다.
증거는 results_archive/completed_v1의 summary.json, frequency_summary.json,
completion_audit.json이다. 원시 imagewise vector와 checkpoint는 공개하지 않는다.

````

</details>
