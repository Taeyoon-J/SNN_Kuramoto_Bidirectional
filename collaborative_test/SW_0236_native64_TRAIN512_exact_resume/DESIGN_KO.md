# SW0236: 같은512회 학습에서 unique TRAIN64와 TRAIN512 비교

SW0226에서는128회 업데이트로 TRAIN64와512를 비교했으나 FG 차이 CI가0을
포함했다. SW0232에서 TRAIN64의 누적512회 평균이128회보다 높아졌다.
따라서 학습량 부족이 데이터 증가 효과를 가렸는지 한 번 비교한다.

실제 SW0226 TRAIN512의128회 encoder·projection·Adam·history·Torch RNG를
직접 복원하고 bitwise 검증한 뒤384회만 추가한다. 누적512회/B16/8192
exposures로 SW0232 TRAIN64와 동일하다. 512개의 고유 이미지는 기존
TRAIN pool에서 고정됐고 앞64장은 SW0232와 같다. 원본 teacher·loss·LR·
clip·나머지 core·graph·gates·mask readout은 유지한다. 반복 노출은
64장128회씩 대512장16회씩이며 신규70,000장 학습이 아니다.

actual accepted128 checkpoint의 CPU 복원/gradient 검사를 통과한 뒤 큐에
등록한다. 235번 세 seed가 먼저 GPU 실행 기회를 받은 후에만236번 GPU
작업을 시작한다. 외부 작업이 있으면 우리 worker만 양보하고 경계에서 재개한다.

고정16장·native64·세 seed를 원래 기준 및 SW0232 TRAIN64/512회와 비교한다.
데이터 증가 효과 확정은 해당 FG 차이의 paired image CI 하한>0일 때만
주장한다. FG 후보 gate는 원래 기준 대비+.01 및 CI 하한>0, TRAIN64보다
FG 평균이 높아야 한다. balanced gate는 두 비교군 모두에 대해 두 IoU
평균도 낮아지지 않아야 한다. 기존16장 GT barrier와 RNG/1024-step spike
mask 계약을 유지한다. 전체320장 기준점 SW0227은 이 미니로 바꾸지 않는다.
