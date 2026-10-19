# SW0232: 수락된128회 checkpoint에서 직접 이어 학습

231번 재계산은 두 seed에서 가중치 약1e-7 수준의 차이로 bitwise128 prefix 기준에
실패했다. 실패 checkpoint/로그를 보존하고 이를 정확한 이어 학습으로 취급하지 않는다.
이번에는 원래223번의 실제 encoder/projection/Adam/history/RNG를 직접 복원한다.
동일성 검증은 완화하지 않으며 복원된128회 상태를 독립 child snapshot으로 보존한다.

loss·TRAIN64·teacher·batch16·LR/clip·graph/나머지 core/gates/readout은 유지한다.
원래128회에서384회 더 업데이트해 누적512회로 진행한다. 새 계산은384 updates/
6144 exposures이고 누적 학습은512 updates/8192 exposures다. 새128회 재계산은 없다.
원래 source로 만든 teacher를 그대로 유지하고, 실제 accepted128 모델의 CPU 복원과
gradient 검사도 GPU 큐 등록 전에 확인한다. 기존 child boundary가 있으면 child
checkpoint/Adam/RNG에서 재개하며 accepted source를 덮어쓰지 않는다.

고정16장·세 seed를 실제223번128회 모델과 비교한다. baseline FG>=+.01 및
paired CI 하한>0,128회 모델 FG 초과를 FG 후보 기준으로 유지하고 두 IoU 평균
비감소를 balanced 기준으로 유지한다. mask-supervised mini이며 전체320장 .90이나
70,000장 재학습 결과로 승격하지 않는다. 이 고정 예산 확인 실패 시 epoch sweep은 없다.
