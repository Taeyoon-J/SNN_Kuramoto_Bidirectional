# SW0231 서버 검증 시작

128회 종료 당시에도 원래 공동학습 loss가 계속 줄어든 기록을 근거로, 같은 loss의
고정512회 업데이트를 한 번 비교한다. loss/데이터/초기 모델/학습률/graph/gates/
readout을 바꾸지 않는다. 같은 TRAIN64·batch16·실제 encoder+projection 학습이다.
TRAIN mask supervision을 사용하며 70,000장 추가 학습이나 무감독 결과는 아니다.

첫128 updates가 실제223번 모델의 encoder/projection/Adam/history/Torch RNG와
같은지 검증하고 snapshot/proof를 보존한다. 다르면 이후 업데이트·native 평가를
진행하지 않는다. 통과하면 같은 Adam 상태에서384 updates를 더 수행한다.
prefix를 다시 계산하는128회 비용도 포함한다. 고정16장·세 seed 평가에서 실제
223번128회 모델과 비교하며, 전체320장 기록은 별도227번이 확인한다.

로컬13개 테스트를 통과한 코드를 배포했다. 서버에서도13개 테스트와
train/evaluate/queue CLI 검사3개를 통과했고 실제 세 seed CPU 검사를 시작했다.
모든 CPU 검사에 통과해야 독점 idle GPU 큐를 등록한다. 이 문서는 서버 검증 시작
시점의 기록이며 GPU 학습 완료나 성능 개선을 주장하지 않는다.

서버 launcher PID2799508, CPU 검사 PID2799562/2799563/2799564.
배포 SHA는 START_EVIDENCE.json, 수락된223번 학습 기록 분석은
TRAIN_LOSS_DIAGNOSIS.json에 기록한다. 동료 브랜치는 이번에도5c88a6f로 변동이
없음을 fetch로 확인했다. 기존227번은 중단·재시작하지 않는다.
