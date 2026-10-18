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
