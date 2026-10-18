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
조회 직후 원래 worker의 실행 상태와 `TracerPid=0`을 확인했으나, 뒤이은 재확인에서
worker에 정지 상태가 남아 있음을 발견했다. 같은 PID에 SIGCONT를 보내 재개했고,
CPU 누적 시간이 다시 증가하며 `TracerPid=0`인 것을 확인했다. 이후 gdb는 사용하지
않고 `py-spy dump --nonblocking`으로만 조회했다. 모델 코드,
checkpoint, 입력, RNG 규칙, readout 및 저장된 mask는 변경하지 않았다.

후속 실행에서 지연을 재현할 경우 native forward와 CPU partition 각각의 경과 시간을
분리해서 기록한다. 현재 표본만으로 readout 변경이나 수치적으로 다른 최적화를
도입하지 않는다. 기존 320장 incumbent 및 원래 16장 gate 실패 기록도 유지한다.

## 메모리 정책 복구

비정지 Python 스택은 `signed_pair_energy`의 `np.log(q_bar/(1-q_bar))`를 반복해서
가리켰다. 별도 합성 진단에서도 4096×4096 float64 배열의 첫 쓰기가 15초 제한을
넘겼다. 해당 진단 프로세스에만 `prctl(PR_SET_THP_DISABLE,1)`을 적용했을 때는
같은 크기 배열 쓰기가 0.097초, 로그 연산이 0.426초에 완료됐다. 작은 1024개
로그 연산과 가상 배열 할당 자체는 기본 환경에서도 즉시 완료됐다. 호스트 가용
메모리는 약 85GiB였고 세션 및 상위 cgroup 메모리 제한은 무제한이었다. 따라서
일반적인 모델 연산량이나 메모리 용량 부족으로 설명하지 않고, transparent huge
page와 관련된 큰 배열 쓰기 지연에 대응한다. 커널 내부의 유일한 근본 원인을
증명했다는 주장은 하지 않는다.

`recover_memory.py`는 기존 worker와 supervisor의 종료를 확인한 후에만 별도 복구
상태 경로를 만든다. 원래 실패 상태, 로그 및 29개 mask의 58개 파일 SHA는 보존했다.
서버 전역 설정을 변경하지 않고 복구 프로세스와 자식 프로세스에만 THP를 끈다.
원래 checkpoint, 입력, fingerprint, GPU 독점 정책, 연산식, RNG 및 저장 경로는
유지한다. 복구 worker는 원래 223번의 offset0 mask와 frequency를 bitwise 재현해야
나머지 평가로 진행한다. 이 추가 replay는 기존 225번의 1회 replay와 별도로 기록한다.

원래 큐는 진단 후 정상 종료 신호로 종료되어 failed 상태를 보존한다. 완료 audit는
복구 큐의 완료, 원래 종료 상태 SHA, protocol SHA 및 정확한 replay 증거를 추가로
검사한다. 로컬 테스트 4개 중 Linux 전용 1개는 skip됐고, Linux에서는 기존 전체
테스트를 포함한 11개가 통과했다. 복구는 추가 학습이나 새로운 성능 결과가 아니다.

첫 복구는 실제 mask/frequency 재현을 통과했으나 같은 interpreter에서 CUDA 초기화 후
원래 predictor의 CPU admission을 재호출하여 종료됐다. 이 실패 및 replay를 v1 경로에
보존했다. v2는 replay 후 같은 PID를 원래 worker의 새 interpreter로 exec해 CPU admission
순서를 지키며, GPU lease와 THP 정책은 유지한다. 해당 전체 실행 경로를 mock으로
검사하고 Linux에서는 exec 이후 실제 THP 정책 유지까지 확인했다. Linux 전체 테스트
12개가 통과했다. v2 등록 전 29개 결과 SHA를 다시 대조했고 이전 프로세스가 모두
종료됐음을 확인했다. v1 실패를 성공이나 새로운 학습 결과로 취급하지 않는다.
