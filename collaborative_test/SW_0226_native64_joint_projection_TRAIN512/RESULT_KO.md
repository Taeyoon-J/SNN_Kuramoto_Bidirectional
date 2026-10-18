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
