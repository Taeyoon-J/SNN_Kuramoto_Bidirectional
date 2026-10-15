# SW0163 운영 오류 및 복구 근거

원래 학습 큐는 `failed_training`으로 종료됐고 예측·평가 큐도 선행 작업 실패로 종료됐다. 성능 평가는 아직 없다. Q_only와 A_only child는 각각512번의 업데이트를 끝내고 `training_complete`로 정상 종료했으며, raw_only와 Q_A는 학습을 시작하지 않았다.

실패는 부모 큐가 여러 스레드에서 산출물을 검증할 때 발생했다. 검증 함수가 기대 초기 모델을 다시 만드는 과정에서 `torch.random.fork_rng`와 전역 `manual_seed`를 사용한다. 이는 스레드 간에 독립적이지 않으므로, 동시에 초기화하면 기대 초기 상태 해시가 달라져 정상 manifest를 잘못 거절한다. 로컬100회 재현 모두 순차 초기화와 불일치했다. 로컬과 서버의 해시 자체를 비교한 결론은 아니다.

서버에서 기존 공개 validator를 한 스레드로 실행한 결과 Q_only/A_only **둘 다 전체 검증을 통과**했다. 두 모델의 초기 상태는 서버의 순차 초기화 값 `c9baacaa239b9ae41d913adc5ed5bee8f505df213c8b4580b24dd89ff9d15470`과 같다. 코드 fingerprint,512-update endpoint, optimizer 및 파일 해시를 보존한다. 이 두 학습을 다시 실행할 필요는 없다.

복구는 새 namespace에서 기존 실패 기록과 종료된 소유 프로세스를 확인한 뒤 두 완료 모델을 채택하고, raw_only/Q_A만 기존 trainer로 실행하는 방식이다. 부모의 모델 생성이 포함된 검증은 전부 직렬화한다. 원래 모델·loss·데이터·학습 순서·업데이트 수·실패 상태 파일은 바꾸지 않는다.

2026-10-14 서버 배포 후 실제 CPU 테스트7개가 skip 없이 통과했고, 기존 public source 준비 및 Q_only/A_only 순차 검증도 통과했다. CUDA는 초기화하지 않았으며 평가 정답은 열지 않았다. TRAIN 정답은 기존 학습 입력 검증 과정에서 재검증했다. 복구 학습/예측/평가 supervisor PID3588458/3588470/3588475를 실행하여 각 실제 `/proc/PID/cmdline`과 초기 상태를 확인했다. 등록 시 학습은 `validating_recovery`, 예측·평가는 새 선행 큐를 기다리는 상태다. 이는 큐 실행 확인이며 학습·성능 평가 완료를 뜻하지 않는다. 증거는 `results_archive/recovery_launch_20261014/server_records.json`에 보존한다.

근거:

- `results_archive/terminal_parent_validation_failure_20261014/server_records.json`: 원본 실패 상태와 완료 endpoint 해시.
- 같은 폴더의 `rng_race_diagnosis.json`: 실제 로컬 재현 범위와 플랫폼 차이.
- `results_archive/sequential_recovery_audit_20261014/receipt.json`: 서버 공개 validator 순차 검증 통과.

Goal 상태와 구분한다. 조회된 goal의 blocked 갱신 시각은1792015170, 서버 학습 큐의 실패 시각은1792016946으로 약30분 뒤다. 따라서 이 서버 오류가 앞선 goal blocked 전환 원인이라고 주장하지 않는다. Goal 도구에는 그 전환의 사유가 제공되지 않았다. 재개 후 goal 상태는 active로 확인됐다.
