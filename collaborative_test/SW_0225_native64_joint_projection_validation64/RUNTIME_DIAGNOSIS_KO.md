# SW0225 저장 지연 진단 — 평가 결과 아님

seed0/1의 64개 mask 생성은 완료됐고 seed2는 29개 저장 후 긴 계산 구간에 들어갔다.
이는 최종 점수나 성능 개선의 증거가 아니다. seed2 worker PID 2633054의 실제
실행 상태와 증가하는 CPU 누적 시간을 확인했다. GPU1에는 해당 worker만 있었고,
조회 시 GPU 사용률은 0%, worker CPU 사용률은 약 99%였다. 오류 로그는 없었다.

한 번의 제한된 native 스택 표본은 NumPy의 `_multiarray_umath` 내부를 가리켰다.
이 표본만으로 정확한 Python 호출 위치나 근본 원인을 단정할 수 없다. 따라서
GPU 연결 실패나 프로세스 종료로 간주하지 않으며, 미완료 mask를 임의로 건너뛰거나
기존 worker를 재시작하지 않았다. 64장·세 seed 전체가 완료되기 전 점수는 발표하지 않는다.

첫 debugger 조회는 제한 시간을 넘겨 해당 진단용 debugger만 종료했다. 이후
외부 debug symbol 조회를 끄고 강제 종료 제한을 추가한 표본 조회는 정상 종료했다.
두 경우 모두 원래 worker의 실행 상태와 `TracerPid=0`을 확인했다. 모델 코드,
checkpoint, 입력, RNG 규칙, readout 및 저장된 mask는 변경하지 않았다.

후속 실행에서 지연을 재현할 경우 native forward와 CPU partition 각각의 경과 시간을
분리해서 기록한다. 현재 표본만으로 readout 변경이나 수치적으로 다른 최적화를
도입하지 않는다. 기존 320장 incumbent 및 원래 16장 gate 실패 기록도 유지한다.
