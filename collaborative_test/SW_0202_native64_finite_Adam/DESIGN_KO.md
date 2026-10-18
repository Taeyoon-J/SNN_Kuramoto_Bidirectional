# SW0202 finite Adam continuation, unchanged QCE

질문: minibatch loss가 감소하지 않는 update를 모두 rollback하는 현재 policy가 native64의
학습을 막는가?199 QCE control과201 count 후보 모두 새 B4 두 proposal이 거절되어
실제 model/Adam이 TRAIN8 boundary2에 머물렀다. 이것이 data scaling 문제의 원인인지는
아직 입증되지 않았지만, loss를 더 추가하기 전에 optimizer 적용 기준을 분리한다.

단일 변경은 finite step을 적용하는 일반적인 Adam 방식이다. 기존 QCE temperature.1,
live64,T1024,settle512, LR5개,clip1, integration parameter projection, graph/CNN/gates/
production readout을 유지한다.201 count loss나200 live512를 섞지 않는다.192의 실제
TRAIN8 model/Adam/RNG boundary2를 이어 indices8..15에서 B4 두 proposal을 추가한다.
같은 시작/데이터/횟수의199 실제 guarded QCE control을 고정 비교군으로 쓴다.

학습 gradient/no-grad full forward의 same-RNG parity와 실제 S/family credit는 계속
검사한다. 유한한 제안은 그 minibatch의 after loss가 증가해도 적용한다. nonfinite
parameter delta/after objective는 model/Adam/RNG 전체를 복원한다. after 평가는 여전히
수행하여 descent 여부를 기록한다. 이 비용을 생략하여 속도 차이를 섞지 않는다.
기존 history의 accepted는 새 proposal에서 **finite step 적용**을 의미한다. 별도
descent_accepted가 이전의 충분한 loss 감소 기준을 보존한다. Adam step 수는 실제
적용 횟수와 일치해야 한다. 원래 최고 checkpoint는 수정하지 않는다.

그룹 개수 정보나 GT는 추론에 제공하지 않는다. 고정 development offsets0,2,...30
16장/RNG193000+offset, canonical2×2 modal GT, 생산 hard mask 및 세 metric. 예측을
고정한 뒤 GT를 열고 metric을 독립 검증한다. paired bootstrap10000(seed199).
확장 gate는 candidate-control FG>=.01 및 CI lower>0, candidate-warm FG mean>=0,
양쪽 objectIoU mean>=0, 새 적용 update>=1, model/CNN committed change다. raw mask
count도 부진단으로만 기록한다. seed0 mini는 full320/three-seed .90 목표 달성이 아니다.
업데이트가 실제로 적용돼도 mask 성능이 개선되지 않으면 이 짧은 recipe는 닫는다.
