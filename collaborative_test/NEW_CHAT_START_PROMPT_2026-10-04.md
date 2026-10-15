# 새 대화 시작 프롬프트

2026-10-15 사용자 정리 범위 확정: **실험 소스는 기존 `collaborative_test`에 유지하고 생성 파일만 이동한다.** 로컬 결과·로그·캐시 실체는 `collaborative_testing/generated/collaborative_test/` 아래에 둔다. 기존 결과 디렉터리는 junction, 개별 결과 파일은 NTFS hard link로 기존 경로와 Git 기록을 보존했다. 새 실험은 로컬 결과를 쓰기 전에 `collaborative_testing/prepare_output.ps1 -Experiment <실험폴더명>`으로 결과 경로를 준비한다. 새 결과는 그 실험의 `results_archive/` 아래에 기록하고 소스 옆에 새 결과 파일을 흩어놓지 않는다. 서버 경로와 실행 중인 서버 작업은 이 로컬 정리로 변경하지 않는다.

2026-10-14 로컬 파일 정리 지시: 배포 준비물·임시 파일·캐시·시각화 산출물은 `patch_v2_sw/collaborative_testing/` 아래에만 생성한다. 현재 private runtime 루트는 `collaborative_testing/collaborative_private_runtime_20261008/`이다. 과거 ACMLab 루트·바탕화면·중첩 ACMLab 경로를 다시 생성하지 않는다. 로컬 테스트는 `collaborative_testing/run_tests.ps1`로 실행하고, pytest 캐시는 `collaborative_testing/cache/pytest`를 사용한다. 이동표와 SHA 검증은 해당 폴더의 organization 기록을 참고한다.

2026-10-14 사용자 운영 지시 추가: goal이 멈추면 실제 상태(pause/blocked), 확인된 원인, 서버 테스트 지속 여부, 재개 조치를 함께 보고한다. 사유가 도구에 없으면 원인 미확인이라고 명시하고 추측하지 않는다. 서버 장애와 goal 상태 전환의 원인을 증거 없이 동일시하지 않는다.

2026-10-14 최신 실험 상태 보완: `collaborative_test/LATEST_STATUS_20261014.md`를 함께 읽고 실제 서버 큐·PID·평가 산출물로 재확인한다. 기존 STATUS/INDEX의 과거 running 문단보다 SW0162–SW0164의 날짜별 기록을 우선한다.

2026-10-14 사용자 운영 지시: 작업 중 SSH 연결 상태와 서버 실험의 실제 PID를 주기적으로 확인한다. SSH에는 `ServerAliveInterval=30`, `ServerAliveCountMax=3`을 적용하고, 정기 확인만을 위한 사용자 메시지는 보내지 않는다. 관찰 명령 종료/시간 초과만으로 학습을 재시작하지 않는다. 연결 유지 설정은 연결 끊김을 완전히 방지하는 보장이 아니며, 실제 접속 오류는 구분해 확인한다.

이 대화는 이전 `patch_v2_sw` 연구 대화의 연속이다. 먼저 다음 인수인계 문서를 전부 읽어라.

`collaborative_test/NEW_CHAT_HANDOFF_2026-10-04.md`

그 후 실제 저장소의 다음 자료를 읽어 문서가 최신 상태인지 검증하라.

- `collaborative_test/STATUS.md`
- `collaborative_test/INDEX.md`
- `collaborative_test/evaluation_contract.md`
- `collaborative_test/SELECTION_POLICY.md`
- `collaborative_test/SW_0033_membrane_signal_ablation/`
- `snn_kuramoto_bidirectional/s2net_cls.py`
- `snn_kuramoto_bidirectional/loss_function.py`
- `snn_kuramoto_bidirectional/evaluation.py`

현재 권위 있는 로컬 저장소는 `patch_v2_sw`, 브랜치도 `patch_v2_sw`다. 마지막으로 확인된 HEAD는 `28d24b0`이지만 반드시 다시 확인하라. 현재 대화의 이전 goal이 자동으로 이전되었다고 가정하지 말고, 다음 목표를 새 goal로 생성한 뒤 계속하라.

목표: 고정 patch 평가 계약에서 실제 spike 또는 membrane history에서 만든 masks의 seed 0, 1, 2 평균이 comparable Slot Attention 평균을 `patch_fg_ari`, `patch_foreground_iou`, `patch_matched_object_iou` 모두에서 엄격히 초과하도록, 진단 기반으로 모델을 반복 개선한다. Core의 기본 계산과 구조는 최대한 유지하고 hyperparameter, loss, classifier를 개선한다. 모든 test를 번호와 재현 정보와 함께 `collaborative_test`에 기록하고 `patch_v2_sw`에 push하며, 동료 `patch_v2`의 새 결과도 평가 계약 차이를 확인한 뒤 유효한 insight만 반영한다.

가장 최신 결론: SW_0033에서 spatial-only spectral readout `0.491272/0.190837/0.178026`과 membrane×spatial `0.494639/0.191070/0.170568`이 거의 같았다. 현재 ARI gain 대부분은 spatial prior이며 membrane object signal은 약하다. 따라서 무작정 classifier sweep을 이어가지 말고, 먼저 theta→sinusoidal gating→dendritic/h-wave→membrane→spike 각 단계에서 same-object/different-object 분리도, AUC, spatial-only 대비 incremental gain, gradient connectivity를 같은 validation subset에서 진단하라. 그 결과로 다음 controlled training을 설계하라.

사용자에게 routine epoch는 보고하지 않는다. major한 모델 변경, 실제 improvement, blocker, 사용자 결정이 필요한 내용만 보고한다. 각 test는 쉬운 말로 무엇을 바꿨고 왜 바꿨는지, 세 지표와 count/foreground tradeoff가 어떻게 변했는지 기록한다. 현재 대화는 삭제하지 않는다.
