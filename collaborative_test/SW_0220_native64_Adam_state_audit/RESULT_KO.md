# SW0220 완료: Adam 상태 초기화만으로 실제 loss 감소를 회복하지 못함

218과 같은 source192 native64 model/Adam step2에서 TRAIN indices8..11의
한 B4 gradient를 구하고 동일한 clipped gradient를 두 arm에 복사했다.
초기 모델/context/RNG/LR/betas/epsilon/projection은 같고, reset arm만 Adam
first/second moments와 counter를 초기화했다. 실제1024-step forward loss다.

| 상태 | 같은 TRAIN4 combined loss |
|---|---:|
| 업데이트 전 | 2.819465 |
| Adam 상태 유지 후 | 2.951871 |
| Adam 상태 초기화 후 | 2.919585 |

초기화 쪽이 유지 쪽보다 낮지만 두 경우 모두 업데이트 전보다 높다.
사전 고정 TRAIN follow-up gate는 false이므로 이 결과를 근거로 reset-Adam
추가 학습을 등록하지 않는다. Beta/LR/update-count micro-rescue도 하지 않는다.
이는 한 source/한 batch의 결과이며 Adam 초기화가 일반적으로 무효라는 증명은 아니다.

두 step의 surrogate-gradient/projected-delta 내적은 각각 -.0007370/-.0010866이었다.
즉 두 경우 모두 surrogate 방향으로는 내려가는 업데이트였지만 실제 binary
full-rollout loss는 증가했다. 이 결과만으로 temporal truncation, hard events,
step 크기 또는 수치 민감성 중 어느 하나가 원인이라고 정하지 않는다.

## 범위와 검증

실제 Adam counters는 유지 arm2->3, reset arm0->1이었다. 동일 gradient의
정확한 복사와 group hyperparameter 불변을 확인했다. 진단 뒤 실제 원래
model/Adam/RNG를 복원하고 transaction digest 및 원래 source SHA와 대조했다.
원래 checkpoint 파일도 변하지 않았다. 임시 step2개, committed update0,
native image-forward16, 고유 TRAIN4, DEV 평가0이다. 새 segmentation 점수나
새로운 trained model/data-scaling/70k 결과가 아니다.

218의 before loss와 네 이미지의 frozen context hashes는 같았다. 다만 유지
arm의 after loss는218의 기록보다 .00468322 높았으므로218 optimizer step의
bitwise 재현이라고 하지 않는다. 이 진단 안의 두 arm은 실제 동일 gradient와
동일 초기 상태를 사용한 비교다. 수치 차이의 원인을 추가로 확정하지 않았다.

Local/server 각8개 tests, 각2개 CLI, 실제 source model/Adam/RNG와 고정 control
CPU admission을 통과했다. 전체 보고서/예외 복원/비유한 step/실제 owner API를
로컬에서 검사한 뒤 전송했다. 실제 GPU1에서617.04초에 완료됐다. 사후 CPU
감사는 실제 saved source content와 보고된 복원 SHA, counters, 판정의 산술을
독립 검산했다. GPU probe를 별도로 다시 실행했다는 주장은 하지 않는다.

Full320 최고 .848615/.592533/.672047과 세 seed FG .90 목표 미달은 유지한다.
Optimizer-state branch는 닫고 다음 모델 시험은 representation/실제 물체 연결의
정보 보존에 근거해 고른다. Naive scalar-Q threshold가 false-merge와 false-split을
동시에 고친다고 가정하지 않는다. 217의 두 오류에서 Q 부호가 반대로 잘못된
경우가 많았고219에서도 pair AUC 향상이 grouping 향상을 보장하지 않았다.

근거: results_archive/completed_v1/report.json 및 completion_audit.json.
