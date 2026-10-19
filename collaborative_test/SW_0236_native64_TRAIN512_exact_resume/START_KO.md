# SW0236 후속 비교 준비 및 실제 CPU 검증 시작

같은512회/B16 업데이트에서 고유 TRAIN64와512의 차이를 확인한다. TRAIN512의 실제 SW0226 128회 checkpoint를 복원하고384회만 추가한다. 원본 teacher·loss·나머지 core/gates/readout은 유지하며, 기존512장 pool을 사용한다. 데이터셋을 교체하거나 새70,000장 학습을 시작하는 것이 아니다.

로컬18개·Linux18개 검사와 CLI3개 검사를 통과했다. 세 seed의 실제 accepted128 checkpoint CPU 복원·gradient 검사를 시작했으며, 통과 후에만 후속 큐가 등록된다. 235번 모든 seed의 첫 GPU 실행 기회가 확인되기 전에는 이 새 학습을 시작하지 않는다. GPU0–3 독점 idle lease와 외부 작업 양보 정책은 유지한다.

목표는 고정16장·세 seed의 실제 spike/membrane mask 비교다. 데이터 증가 효과는 TRAIN64/512회 대비 FG 차이 CI 하한이0보다 클 때만 확정한다. 미니 결과로 전체320장 최고나 .90 목표를 갱신하지 않는다. 실제 시작 단계와 소스 SHA는 START_EVIDENCE.json에 기록한다.
