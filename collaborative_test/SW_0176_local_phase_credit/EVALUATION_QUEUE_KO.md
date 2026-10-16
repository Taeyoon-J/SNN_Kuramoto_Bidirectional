# SW176 자동 평가 등록

64회 업데이트 학습이 끝나면 동일한 고정 32×32 patch / 320장 / seed1 평가를 실행한다. 예측 부모 PID 228242, CPU 채점 부모 PID 228264의 실제 명령행과 대기 상태를 확인했다. 서버 CPU 검사 10개와 실제 대조군 예측 파일 일치 검사를 통과했다.

학습 종료 전이므로 최종 모델 CPU 로드는 아직 수행하지 않았다. 예측 부모가 완료된 학습 manifest와 checkpoint를 검증하고 실제 모델을 CPU에서 로드한 후에만 비어 있는 GPU를 확보한다. 기존 학습은 재시작하지 않는다.

후보의 conditional 및 standalone 예측을 모두 확정한 뒤 정답 mask를 읽고 세 지표를 계산한다. SW164 원본과 SW175 critical-component 대조군은 보존된 같은 320장의 예측을 사용하며 다시 학습하지 않는다. 각 지표와 bootstrap 비교를 재계산해 검증한다.

이 실험은 TRAIN512 두 번 노출, seed1의 짧은 supervised 비교다. 70,000장 새 학습 또는 세 seed 최종 성과로 해석하지 않는다. 최종 성능은 아직 미정이며 최고 모델은 변경하지 않았다.

## 실제 평가 시작 확인

64-update 학습이 완료됐고, 최종 checkpoint를 CPU에서 실제 로드하여 manifest와 state SHA가 일치함을 확인했다. 서버 CPU 검사 10개도 통과했다. GPU1의 실제 prediction worker PID 237914가 부모 228242 아래 실행 중이며 scorer 228264는 예측 확정을 기다린다. 최종 점수는 아직 없고 최고 모델은 유지한다. 증거는 `results_archive/training_complete_evaluation_started_20261016/`에 보존했다.

## 채점 foreground 검사 수정

Root 재검토에서 v1 채점 core가 conditional뿐 아니라 standalone도 Source-T foreground와 동일해야 한다고 잘못 검사한 것을 발견했다. 등록한 standalone 출력은 native QCC support를 쓰므로 이 조건은 적용할 수 없다. 기존 예측의 `_read_row`는 standalone mask가 저장된 실제 QCC support와 일치하는지 이미 검증한다.

정답을 읽기 전 대기 중인 CPU 채점기 228264만 의도적으로 중지했다. 이로 인한 v1 queue의 failed 상태와 중지 전후 명령행·상태를 보존했다. 원본 학습·예측·채점 코드 파일을 수정하지 않고 `score_v2.py`, `score_queue_v2.py`, `test_score_v2.py`를 새로 배포했다. 조건부 Source-T foreground 검사는 유지하고 standalone에는 등록한 native-QCC 계약을 적용한다. metric 계산, 대조군 재계산, bootstrap 및 확대 조건은 동일하다.

서로 다른 conditional/standalone foreground를 가진 320장 synthetic HDF5 검사, 잘못된 conditional의 GT 접근 전 거부, 기존 v1 prediction 큐 인수인계를 포함해 로컬·서버 각각 13개 검사가 통과했다. 수정 scorer PID 245039가 실제 대기 중이다. 새 score 경로는 `results_archive/evaluation_seed1_v2_20261016.json`이다. 학습과 GPU 예측은 재시작하지 않았으며 최종 성능은 아직 미정이다.
