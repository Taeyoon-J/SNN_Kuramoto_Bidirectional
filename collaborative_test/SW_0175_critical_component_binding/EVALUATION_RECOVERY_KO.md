# SW175 학습 완료와 고정 320장 평가 복구

native_Q 대조군과 critical_component 후보 모두 seed 1의 **64 optimizer updates를 완료**했다. 두 최종 checkpoint와 optimizer/boundary, 원본 학습 소스는 보존했다. 평가 복구에 따른 추가 학습은 없다.

이전 평가 큐는 입력 검증에서 종료됐다. 평가 코드가 source-cache manifest SHA를 학습 binding의 최상위에서 읽었으나 실제 train_v2 생산자는 `bindings.source_proof.source_cache_manifest_sha256`에 기록한다. 실제 캐시 변경은 없었고 두 모델 모두 중첩 provenance와 현재 캐시 SHA가 일치한다. 예측 및 VAL 정답 채점 전 실패했으므로 그 큐의 성능 점수는 없다.

수정은 새 `predict_evaluate_v2.py`, `prediction_queue_v3.py`, `score_queue_v3.py`와 새 출력 경로에 적용했다. 기존 실패 버전과 로그를 보존했다. 새 코드는 두 모델의 중첩 SHA를 모두 검증하며 검증을 생략하지 않는다. 로컬·서버에서 각 11개 CPU 계약 검사를 통과했다. 특히 서버에서는 **완료된 실제 두 checkpoint로 전체 prepare_prediction_inputs_cpu 및 strict 모델 로드, 최종 state SHA, 고정 encoder/전처리/geodesic 구성 일치를 확인한 후** GPU 평가를 등록했다. CPU 검증에서 VAL mask는 읽지 않았다.

고정 native32 320장에 대해 conditional Source-T foreground와 standalone learned-Q foreground의 두 endpoint를 각각 생성한다. 두 모델의 전체 예측을 확정한 뒤에만 정답 mask를 열고 세 지표 및 사전 등록된 paired bootstrap을 계산한다. 후보의 두 endpoint 모두 기존 SW164 seed1과 대조군을 넘는 등록 조건을 만족해야 다른 seed로 확장한다. 현재는 최종 평가 대기이며 최고 모델 SW165를 교체하지 않는다.

실제 완료 학습 manifest, TRAIN8 진단, 이전 큐 terminal 증거, 수정 소스 SHA, 전체 실제 CPU 로드 결과 및 새 PID 관측은 `results_archive/completed_training_evaluation_recovery_20261015/`에 기록했다. TRAIN8은 학습 진단이며 검증 성능으로 사용하지 않는다.
