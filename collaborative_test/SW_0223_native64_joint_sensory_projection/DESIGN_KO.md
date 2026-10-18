# SW0223: encoder와 실제 core 감각 projection 공동 학습

222번은 내부 주파수 분산을 줄였지만 FG 평균 +.0114의 CI가0을 포함했고
Object IoU가 하락했다. 이번에는 동일 source/teacher/TRAIN64/128 updates/B16/
encoder LR/목표함수를 유지하고 실제 gamma_channel_proj.weight(4x8,32개)를
함께 학습한다. projection LR은 기존 context LR3e-4로 고정한다. omega,
frequency gain, phase gain, 나머지 graph/gate/core 파라미터는 모두 고정한다.
추가 projection을 끼우거나 채널 순열을 DEV에서 고르지 않는다.

고정 teacher는 원래 source로 먼저 계산한다. gamma anchor와 자연 물체별
주파수 중심 목표는 그대로다. 전체 학습 파라미터에 global norm clip1을 쓴다.
gradient와 실제 projection 변경을 기록하고, 나머지 core 상태의 바이트 SHA가
유지되는지 확인한다. 작은 projection+encoder만 매 update checkpoint에 저장한다.
이는 smooth 감각 입력 pretraining이며 hard spike/gate loss의 end-to-end 학습이나
각 gate의 인과 기여 증명으로 취급하지 않는다.

seed0/1/2 공통 recipe. 고정 DEV16/IDs1320,1322,...1350/canonical128/no crop/
2px modal GT/RNG193000+offset/1024 settle512 기존 spike readout을 유지한다.
세 candidate protocol 및 기존222 control mask를 모두 검증한 뒤 GT를 읽는다.
222 encoder-only와 기존 source를 함께 비교하며 control protocol SHA를 CPU
admission에서 미리 고정한다. 개별 seed를 다른 recipe로 골라 평균내지 않는다.

사용자의 FG 우선순위와 SELECTION_POLICY의 FG/balanced track을 구별한다.
FG track full320 확대 후보 조건은 source 대비 FG+.01/paired image CI하한>0,
그리고 FG 평균이222보다 높음이다. IoU tradeoff를 반드시 같이 기록한다.
Balanced gate는 여기에 source 대비 두 IoU 비감소를 요구한다. 이는 사전 후보
선정 규칙이며 최종 성공은 아니다. 최종 full320 세 seed 평균은 동일 Slot 계약의
세 지표를 모두 엄격히 넘고 최신 사용자 목표인 FG .90도 검증해야 한다.
221/222의 기존 gate 실패는 변경하지 않는다. 새70k 학습은 하지 않는다.
