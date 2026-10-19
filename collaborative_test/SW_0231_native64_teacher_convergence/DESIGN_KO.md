# SW0231: 검증된 원래 loss의 학습 예산 확인

228/230번 간격 보존 loss는 종료한다. 223번 원래 teacher denoising/gamma anchor,
초기 모델·TRAIN64·순서·LR/clip·batch16·encoder+실제 core projection 학습을
유지하고 고정 업데이트 수만128에서512로 늘린다. graph/나머지 core/gates/readout은
고정하며 TRAIN mask supervision을 사용한다. 전체70,000장 재학습은 아니다.

223번 실제128-update history에서 마지막8 epoch의 평균 TRAIN loss가
각각 약4.10%/1.92%/1.10% 감소했다. 이는 아직 objective가 줄어든다는 근거이며
최종 mask 성능이나 데이터 scaling 개선의 증명은 아니다. 추가 예산이 실제
generalization에 도움이 되는지 한 번만 확인한다. DEV로 최적 step을 선택하거나
여러 epoch/LR를 sweep하지 않는다.

첫128 updates를 실제 재현하고 encoder/projection/Adam/history/Torch RNG가
수락된223번 checkpoint와 bitwise 같은지 확인한 후384 updates를 더 진행한다.
동일성 실패 시 추가 업데이트와 native mask 평가를 진행하지 않는다. 실제128회
snapshot/proof를 별도 보존한다. resume 시 검증된 checkpoint/Adam/RNG를 복원한다.
수치상 같은128회 prefix를 재현하는 계산은 포함되므로 추가 계산 비용을 숨기지 않는다.

평가는 기존 고정16장·세 seed·원래 spike mask/RNG로 진행하며 실제223번128회 모델을
대조한다. baseline FG>=+.01/paired CI 하한>0 및128회 모델 초과를 FG 후보 기준,
FIoU/OIoU 평균 비감소를 balanced 기준으로 유지한다. 모든 결과는 mini로 기록하고
전체320장·.90 달성으로 승격하지 않는다. 실패 시 이 예산 확인 분기는 종료한다.
