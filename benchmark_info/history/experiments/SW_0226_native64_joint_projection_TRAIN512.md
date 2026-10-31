# SW_0226_native64_joint_projection_TRAIN512

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `e7d2bbf5eba83ae7d95da5202ad08b7f8487d7a7e5bf44a2dc925504f6526a8b`

````markdown
# SW0226: 공동학습의 고유 TRAIN64 → TRAIN512 비교

223번과 같은 원래 seed0/1/2 checkpoint에서 시작한다. mask-supervised 고정 source
object-frequency teacher와 gamma anchor, encoder 전체 및 실제 4×8 core projection
학습을 유지한다. 다른 core/graph/gate는 고정한다. TRAIN pool의 첫512개 중 64개만
반복한 223번과 비교하며 128 Adam updates, batch16, 총2048 exposures, LR/clip은 같다.
512장은 64장의 정확한 prefix를 포함하고 그 prefix RGB/GT/teacher SHA를 검증한다.
초기 모델과 DEV16 RGB/baseline도 동일해야 한다. 221번 unit-margin loss 재시도가 아니다.

223번 TRAIN64의 학습과 mask는 재사용한다. 새로운 TRAIN512 세 seed의 최종 mask를
모두 동결한 후 IDs1320,1322,..1350의 원래 native64 modal GT를 읽는다. 1024 steps,
settle512 및 원래 spike/membrane readout을 유지한다. paired image bootstrap은 기존
recipe와 같은 RNG221/10000회이다. full320 후보 기준은 source 대비 FG≥.01, CI하한>0,
TRAIN64보다 높은 FG이고, balanced 후보는 두 IoU도 source 대비 비감소여야 한다.
별도로 TRAIN512−TRAIN64 FG 차이 CI하한>0일 때에만 이 고정모델·미니범위의 data
scaling 증거로 기록한다. seed 모집단, 70000장 scaling, untouched test의 증명이 아니다.
전체 목표는 full320에서 세 seed 평균 all3 Slot 초과와 FG .90이며 mini로 축소하지 않는다.

225번에서 진단된 THP 지연을 피하도록 Linux 큐/worker에만 huge page를 끈다.
전역 설정과 연산식은 변경하지 않는다. GPU0..3 중 다른 compute 소유자가 없는
장치만 공유 lease로 예약하고 외부 작업이 나타나면 본인 worker만 yield한다.

````

</details>

## PEER_REVIEW_KO.md

<details>
<summary>PEER_REVIEW_KO.md</summary>

Source SHA256: `11f296f3ad7205c4b08e65de97a1b2d9f6a9222faeee96d3e224b32b861e11d8`

````markdown
# 동료 브랜치 확인

이번 배포 전 `origin/patch_v2`를 fetch했고 최신 commit은
`7a21632b93805f4f1da69a2e77a9ddd95c624d0a`로 이전 확인 이후 변경이 없었다.
해당 commit은 32×32의 8-epoch 개선 주장이 세 seed에서 유지되지 않아 철회됐다는
기록이다. 새로운 native64 구조·loss 개선을 가져온 것으로 취급하지 않는다.
이번 비교도 세 seed 및 같은 데이터·readout을 유지하고, 작은 validation의 개선을
full320 결과나 seed 모집단의 개선으로 확대 해석하지 않는다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `1df522655f856934f5b2f9ee3490a4fc545dda73a0f619f537c773bcb0e4d719`

````markdown
# SW0226 COMPLETE

Fixed16/three-seed TRAIN512 .860512/.650250/.712453 vs TRAIN64 .859634/.645611/.706340.
All paired data-size CIs include zero; no confirmed data-scaling gain.
Source-relative candidate gate passed; full320 incumbent unchanged. See RESULT_KO.md.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `553d51d2df70e2c69c3ec8a61aece45819d6e97ae8f48705bf5546bb9137cb45`

````markdown
# SW0226 완료: 고유 TRAIN64→512, 같은128 updates

같은 초기 모델·encoder/core projection 공동학습·loss·LR·batch16을 유지하고,
고유 TRAIN 이미지만64→512로 늘렸다. 두 조건 모두128 Adam updates/2048 exposures다.
64장은 이미지당32회,512장은4회씩 노출된다. TRAIN64의 prefix RGB/GT/teacher 및
원래 checkpoint와 DEV16을 실제 검증했다. 평가는 native64, IDs1320,1322,..1350의
고정16장·세 seed다. 225번64장 평균 .8772와 직접 비교하지 않는다.

| 조건 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 원래 모델 | .842891 | .600692 | .702161 |
| TRAIN64 공동학습(223) | .859634 | .645611 | .706340 |
| TRAIN512 공동학습(226) | **.860512** | **.650250** | **.712453** |

TRAIN512−TRAIN64의 FG 차이는+.000879, paired95% CI[-.000427,.002702]다.
FIoU 차이+.004639의 CI[-.035953,.037542], Object IoU 차이+.006113의
CI[-.004803,.018197]도0을 포함한다. 평균은 모두 소폭 높지만 **데이터 증가가
성능을 확실히 높였다고 결론 내리지 않는다.** 반대로 더 많은 데이터가 나쁘다는
증거도 아니다. 같은 업데이트 예산의 특정 recipe/고정모델/작은 반복 validation에
대한 결과이며 seed 모집단의 CI,70,000장 scaling 또는 더 긴 학습의 결과가 아니다.

원래 모델 대비 FG 차이는+.017622 CI[.000410,.041731]이고 FIoU도 개선됐다.
사전 등록된 FG/balanced 후보 기준은 통과했지만, TRAIN64 대비 data scaling 기준은
통과하지 못했다. 이 두 판정을 구분해 기록한다. 이 결과만으로70,000장 재학습이나
동일 recipe의 장시간 학습을 시작하지 않는다. 모델은 작은 FG/IoU 개선 후보로 보존한다.

물체 내부 실제 구동 주파수 분산은 원래 모델 대비 세 seed에서 약22–33% 줄었다.
TRAIN64와 비교하면 거의 같거나 조금 높았고, 인접 물체 중심 거리도 원래 모델보다
줄어드는 현상이 유지됐다. source object-center MSE는 .000172/.001078/.000527이다.
주파수 응집의 유지가 모든 gating의 인과 기여나 최종 mask 개선의 충분조건을
증명하지 않는다. TRAIN 데이터가 달라 초기 분산도 다르므로 두 TRAIN loss의 절대
크기를 직접 성능 차이로 해석하지 않는다.

세 실제128-update checkpoint의 encoder·32-weight projection·Adam·frozen core,
teacher/control 및 모든protocol을 검증했다. 총144개 mask를 독립 재계산했고,
원래 metric 구현과의 차이 및 로컬 평균/CI 차이는1e-12 미만이었다. 추가 GPU
forward와 재학습은0회다. 전체320장 incumbent는210번 그대로이며,227번은 별도로
검증된223/225 고정 모델을 전체320장에서 확인 중이다.

근거: results_archive/completed_v1/{summary.json,completion_audit.json,archive_receipt.json}.

````

</details>

## START_KO.md

<details>
<summary>START_KO.md</summary>

Source SHA256: `9207cc02d237010e7fedaf79565934ade2a8fd4c1c4de2769bb4b5f98c5409c1`

````markdown
# SW0226 시작: 공동학습 TRAIN512 비교

로컬 10개·Linux 10개 테스트와 3개 CLI 검사를 통과했다. 세 seed 모두 실제 CPU
admission 및 projection gradient 검사를 통과했다. 원래 source teacher의 초기
within-object variance와의 최대 차이는 4.37e-10이고, TRAIN64 prefix의 RGB/GT/
teacher와 원래 223번 checkpoint/protocol/mask를 검증했다. 세 TRAIN512 큐가 등록됐고
관측 당시 seed1 worker가 GPU3에 배정됐으며 나머지 supervisor도 살아 있었다.
이 기록은 모델 성능 결과나 optimizer 완료 기록이 아니다.

encoder 전체와 실제 core 입력 projection을 같은 128회 업데이트로 학습하며,
고유 이미지 수만 223번의 64장 대신 512장으로 늘린다. source, loss, LR, batch16,
readout, DEV16 및 세 seed를 유지한다. 총2048 image exposures로 예산을 맞춘다.
새 TRAIN512의 최종 mask를 모두 동결한 뒤 기존 TRAIN64와 비교한다. 70,000장 학습이나
full320 확인은 이번 미니에 포함하지 않으며 최고 full320 기록도 갱신하지 않는다.

225번은 별도로 계속 실행 중이다. Linux 큐/worker에는 큰 배열 쓰기의 THP 지연을
피하도록 프로세스별 메모리 정책을 적용한다. 전역 설정과 모델 계산식은 바꾸지 않는다.
빈 GPU0..3만 독점 lease로 사용하며 다른 compute 작업이 나타나면 본인 worker만 양보한다.

## 후속 실제 GPU 학습 확인

server_unix1792363053에서 seed1 worker PID2665968가 GPU3에서 실제 encoder 및
core projection 128 Adam updates를 완료한 training_report를 확인했다. 다른 core
업데이트는 0이고, 이후 원래 spike readout 평가를 실행 중이다. 이는 최종 평가
점수가 아니며 나머지 두 seed는 빈 GPU를 기다리는 큐에 있다. 근거는
results_archive/deployment_v1/gpu_training.json이다.

````

</details>
