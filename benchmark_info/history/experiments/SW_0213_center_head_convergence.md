# SW_0213_center_head_convergence

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `dc3550b13096ff373c2992563b7853683e430b0b1555c45c45d6fa80d671d840`

````markdown
# SW0213: center head의 TRAIN 비작동을 단일 예산 확장으로 검증

211/212 실패 이후의 별도 실험이다. 211은 실제 TRAIN32에서도 center
검출 fallback100%, GT center의 .3 도달률0이었다. 반면 offset 오차는
무변위4.2434→실제입력1.5185 patch로 줄었다. 이 진단에 근거해, sparse
center 검출기가 작동할 만큼 학습됐는지 한 번만 추가 확인한다.

동일211 head/Adam/RNG128을 복사해 총512까지384 update를 추가한다.
원래211 checkpoint와128-step 실패 판정은 보존한다. TRAIN32/고정16/seed2,
loss·LR3e-4·width32·threshold .3 및 .5·capacity10은 모두 그대로다.
Core/encoder/gates는 동결하고 기존268-channel cache를 사용한다.

Primary는512 원래전경 보존 readout이다. 학습전경 readout은 별도 secondary,
TRAIN-mean/moments-only 입력 controls도 등록한다. 모든 endpoint를 먼저
고정하고 GT를 읽으며 validation에서 예산/threshold를 고르지 않는다.
TRAIN center 검출과 실제 Adam512/원본 history128 보존도 검증한다.

주요 구현은 기존211 training library를 그대로 사용한다. 새 child boundary의
steps512는 명시적으로 등록하고, 별도 origin에 부모128 boundary SHA와
bindings를 남긴다. Library fingerprint는211 dependency identity이며213의
전체 science fingerprint는 별도 admission/예측 protocol에 고정한다.

이 한 번의 예산 확장으로도 validation grouping 개선이 없으면 예산을
1024/2048로 반복 확대하지 않는다. Core forward0인 classifier 수렴 실험이며
새70k/전체320/세seed .90 혹은 모든 gating의 인과 기여 증거가 아니다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `f02c12701a583436e0355d69397f8d55666a65e598df69ac7f6c5a89e258dac1`

````markdown
# SW0213 single center-head convergence check

211/212 failed with one-center fallback even on TRAIN. Offset learning was
present (TRAIN error1.5185 versus zero-offset4.2434); center detection was not.
Continue the exact parent128 head/Adam/RNG to one fixed512 endpoint, using cached
TRAIN32/native64 features. Same loss, LR, width and readout thresholds. No core
forwards or core/encoder updates. See DESIGN_KO.md.

Primary512 preserves original spike foreground. Learned foreground and two
frozen-input controls are secondary endpoints. DEV GT opens after all masks
validate. Source211 and its failed128 verdict are preserved. This is a separate
posthoc-motivated convergence experiment, not a relabeling of211 success.

Local continuation tests verify exact parity with uninterrupted Adam, original
parent bytes/history preservation, resumed endpoint identity and changed-data
rejection. The initial local runtime-guard test caught a missing deterministic
cuDNN context; it was corrected before deployment and all five local tests pass.
Linux tests, actual parent/native64 CPU gradient admission and queue receipts
will be recorded under results_archive before model execution.

Five local/five Linux tests, actual parent128/native64 CPU gradient and saved
software/math-setting admission passed before launch. Supervisor2347302,
GPU1 worker2347432 completed the fixed512 continuation. Actual Adam512 and
original128 history/source bytes plus96 mask endpoints independently audited.
Primary .589666/.445474/.325129 versus original .788414/.445474/.674335:
no gain. TRAIN center recall98.77%, fallback0; extending budget fixed center
detection but not grouping. Count exact4/16 is retained as a limited pilot.
No further budget/threshold sweep. See RESULT_KO.md. Full320 incumbent unchanged.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `4b7abc50c28360590c0f4b0ab9c28a736e9d154f36d098e0338ea98d0bd891f1`

````markdown
# SW0213: center 검출 수렴은 회복, 원래 grouping 대비 성능 개선 실패

211의 실제128 head/Adam/RNG에서 총512까지384 update를 추가했다. 동일
TRAIN32/seed2/native64/cache, 동일 loss/LR/width/threshold를 사용했다.
원래 부모 checkpoint와 history128을 보존하고 독립 검증했다. 추가 core
forward/encoder update는0이며 전체320/70k 학습 실험이 아니다.

| 고정16 설정 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 warm205 core mask | .788414 | .445474 | .674335 |
| 부모 center head128 | .034678 | .445474 | .044414 |
| **primary512** | **.589666** | **.445474** | **.325129** |
| frozen TRAIN-mean input | .034678 | .445474 | .044421 |
| frozen moments-only input | .254848 | .445474 | .126680 |
| secondary 학습 전경 | .593320 | .266504 | .189443 |

실제 Adam512와384개 추가 update, source 원본 보존,96개 mask endpoint,
세 지표와 paired CI를 CPU에서 독립 재검증했다. TRAIN 정답 center의
고정 .3 도달률은98.77%, TRAIN 및 primary DEV의 fallback 비율은0이었다.
따라서128의 center 비작동은 해결됐지만, 학습된 center/offset로 원래
물체별 assignment를 대체하는 방식은 여전히 더 나쁘다. 단순히128에서
학습이 전혀 없었다고 하거나,512로 더 학습하면 기존보다 좋아졌다고 하지
않는다. TRAIN center recall과 DEV FG-ARI는 서로 다른 지표다.

실제 입력은 frozen mean/moments-only 입력보다 높은 FG-ARI를 냈다. 이는
이 head에서 관계 특징이 유용하다는 제한된 단서다. 모든 gating의 물리적
인과 기여나 기존 readout 대비 개선을 입증한 것은 아니다.

Count는 원래 평균21.8125에서5.625로 바뀌었고 정답 평균은5.375였다.
개수 exact accuracy는4/16=25%였다. 작은 seed2 pilot의 count 후보로
checkpoint를 보존하지만 안정적인 object-count classifier라고 부르지 않는다.
학습 전경 secondary도 count exact25%이며 두 IoU는 더 낮았다.

등록한 단일 예산 확장을 마쳤다. 이 head를1024/2048로 계속 학습하거나
DEV threshold/LR/width를 sweep하지 않는다. 기존 .848615 전체320·세seed
기준점을 유지한다. 다음 방향은 core의 좋은 기존 partition을 보존하고
분리된 component 간 같은 물체 관계를 학습하는 보정 classifier다.

증빙: results_archive/completed_v1의 summary, origin, training/snapshot 보고,
GPU attempt ledger 및 completion_audit.json. GPU1 worker2347432가 실행됐고
학습이 짧아 live NVIDIA 시작 관찰 전에 완료됐다. 저장된 실제 CUDA
checkpoint와 Adam/history/실행 기록을 검증했으며 live 관찰을 했다고
주장하지 않는다. 로컬5개/Linux5개 테스트와 실제 부모/native64 CPU gradient,
부모 software/math 설정 사전검증은 실행 전에 통과했다.

````

</details>
