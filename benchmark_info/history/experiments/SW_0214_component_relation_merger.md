# SW_0214_component_relation_merger

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `df435aea86ee0df97cd1ef0810d0d7227d65f3e238890ed5b932db2675547e6c`

````markdown
# SW0214: 기존 core component를 보존하는 관계 보정

211–213은 모든 patch의 instance assignment를 center head로 대체해 기존보다
나빴다. 기존 source205 seed2의 좋은 내부 partition은 보존하고, 같은 물체로
판단한 원래 component끼리만 합친다. 기존 component를 쪼개거나 전경을
삭제·추가하지 않는다. 이 방식은 기존 merge 오류를 고칠 수 없다는 한계도
명시한다. 기준은 동일 고정16/native64, source FG-ARI .788414다.

동일211 cache의 TRAIN32/DEV16/268-channel actual spike-rho/membrane 특징을
재사용한다. Core/encoder/gates 업데이트와 추가 core forward는0이다.
Component는 첫 occupied patch의 공간 순서로 정렬해 label ID 순서와 독립화한다.
TRAIN-only channel normalization 후 component 평균268과 위치·크기·채움률6을
계산한다. Pair 입력은 대칭 abs-difference/product: 특징536+geometry12=548.
RGB/GT count/DEV GT는 입력에 없다.

TRAIN target은 두 component의 정답 object histogram 내적이다. Histogram은
각 component의 전체 patch 수로 나누고 background0을 제외한다. Mixed
component는 soft target을 가지며 background끼리를 같은 물체로 학습하지
않는다. Image마다 전체 pair weight를1로 맞추고, positive/negative soft mass를
각각 normalize한 BCE를 사용한다.

MLP548→64 SiLU→1, 마지막 weight/bias0으로 시작한다. Strict logit>0만 merge
후보라 초기 readout은 기존 mask와 bitwise 같다. Adam1e-3, clip1,
full TRAIN-pair batch128 update 한 endpoint를 고정한다. Sparse center 검출이나
물체 개수 고정은 없다. 학습 및 Adam/RNG를 매 update 저장하고 재개한다.

Inference는 complete-link agglomeration이다. 모든 cross-component pair logit이
양수인 두 그룹만 합치며 그 최솟값이 가장 큰 후보부터 처리한다. Tie는 공간
canonical 순서로 결정한다. 따라서 약한 A–C 관계를 A–B–C chaining으로
우회하지 않는다. Validation threshold/count sweep은 없다.

Primary와 frozen geometry-only control을 먼저 고정한 뒤 GT를 읽는다.
Control은 첫536 입력을 TRAIN-pair mean으로 대체하고 geometry12는 유지한다.
이는 재학습한 물리적 gating ablation이 아니다. 모든 세 지표·count·merge 수와
paired CI를 보고하고, FG 우선 후보도 보존한다. 확대 gate는 FG+.01 이상,
paired95% CI lower>0, ObjectIoU 감소 없음이다. 실패 시 작은 계수 sweep 대신
component를 합치는 것만으로 충분한지 진단하고 방향을 판단한다.

Supervised readout pilot이며 새로운 전체320/세seed .90 또는70k 학습 결과로
부르지 않는다. 기존 전체평가 .848615 기준점은 따로 보존한다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `9a179c2a98e73847a36f176876827e7ed233b5e9ea08e0b1394216b3d0679d6e`

````markdown
# SW0214 source-component relation correction

Keep the original core partition and learn which of its components to merge.
No patch reassignment within a source component, foreground change, center
detector or GT object count at inference. Cached actual spike/membrane features
from211 are reused; failed211–213 classifier weights are not used.

One full-batch128 Adam endpoint, symmetric548-dimensional pair features,
soft TRAIN object-overlap targets and complete-link merging. Zero output is
exactly the old mask. Geometry-only is a frozen input control. See DESIGN_KO.md.
Native64/seed2/TRAIN32/DEV16 supervised pilot; full320 .848615 reference preserved.

Eight local/eight Linux tests and actual6648-pair/native64 CPU admission passed.
Supervisor2350264, actual GPU1 Adam128 completed. Independent48-endpoint audit
passed. Primary .720607/.445474/.580461 versus original .788414/.445474/.674335:
gate failed. Frozen geometry .814008/.445474/.632410 has FG CI crossing0 and lower
ObjectIoU, so is an unconfirmed FG candidate, not a new best. See RESULT_KO.md.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `6cf014f7cd5e6c0eeae965adb451f8101e3a2c3c1697c1c46e69527649dafe18`

````markdown
# SW0214: component 보정 classifier의 primary 개선 실패

Native64·고정16·seed2, 기존211 cache/TRAIN32를 사용했다. 새 core forward와
core/encoder update는0이다. Local8/Linux8 테스트, 실제6648 TRAIN pair의
finite gradient와 zero-edge readout의 정확한 baseline 재현을 사전검증했다.
GPU1에서 full-pair Adam128을 완료했고48개 mask·TRAIN normalization·Adam과
평가값을 독립 재검증했다. 기존 component 내부와 전경은 그대로 보존됐다.

| 설정 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 core | .788414 | .445474 | .674335 |
| primary 관계 특징+geometry | .720607 | .445474 | .580461 |
| frozen geometry-only control | .814008 | .445474 | .632410 |

Primary FG 차이는 -.067807, paired95% CI[-.139363,-.007868]이다.
Geometry control의 FG 차이+.025594는 CI[-.013878,+.085607]로0을 포함한다.
ObjectIoU 차이는 -.041925, CI[-.084533,-.002544]다. 따라서 .8140을 확정된
개선이나 새로운 전체평가 최고점으로 부르지 않는다. FG 후보 기록은 보존한다.
Control은 같은 trained head의 입력만 가린 OOD 비교이며, geometry-only로
재학습한 모델이나 물리적 gating ablation이 아니다.

TRAIN relation loss는 .693147→.144341로 감소했다. Primary는 평균9.3125개
component를 합쳐12.5개를 예측했고, control은15개를 합쳐6.8125개를 예측했다.
정답 평균은5.375개다. 단순히 component 수를 줄이는 것이 정확한 grouping을
보장하지 않는다.

후속 점검에서 positive/negative mass를 각각 normalize한 loss와 raw logit>0
판정의 의미가 다름을 확인했다.215는 기존214 판정을 보존한 채 TRAIN에서만
도출한 log-odds shift를 적용하는 별도 비교다. DEV threshold sweep은 하지
않는다. 많은 anchor 특징의 위치 의존성/일반화 문제도 가설로 남지만 frozen
control 결과만으로 원인이라고 확정하지 않는다.

기존 전체320·세seed .848615/.592533/.672047 기준점을 유지한다.
증빙: results_archive/completed_v1/score_v1/summary.json과 completion_audit.json,
deployment_v1/registration_receipt.json과 GPU 실행 ledger. 짧은 GPU 실행은
live NVIDIA 관찰 전에 완료됐으며 실제 CUDA checkpoint/Adam 기록을 검증했다.

````

</details>
