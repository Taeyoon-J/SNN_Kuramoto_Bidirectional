# SW176 local phase gradient: 시작 기록

긴 Kuramoto 재귀를 통한 학습 신호 전달만 바꾸는 실험이다. 자세한 변경·대조·확대 조건은 DESIGN_KO.md에 사전 기록했다. 모델의 forward 연산, native component spike, Q/A, threshold와 mask readout은 바꾸지 않는다. D/M/event/reset의 장기 gradient는 여전히 남아 있다.

현재 실행 경로는 `python -m collaborative_test.SW_0176_local_phase_credit.resource_queue_v2`이다. 이 단계는 TRAIN 위치0의 이미지 한 장에서 no-update source 검증 및 full1024 gradient를 측정하는 실험이다. 64-update 학습이나 검증 성능 결과가 아니다.

로컬·서버 각 15개 CPU 검사와 실제 TRAIN 캐시 준비, fresh source+context 모델 생성을 통과했다. 생성한 초기 state SHA `f372b3a450ebeaf0d49e20d019d3c0676d4998cc1c49586e4c4e4c89e969d7f5`가 완료 SW175 critical-component 대조군의 초기 state SHA와 정확히 일치했다. 이후 GPU1 worker의 실제 PID를 확인했다. 결과 대기 중이며 기존 최고 모델은 유지한다.

최초 v1의 실제 CPU 모델 생성 검증은 context 초기화 SHA를 잘못된 메타데이터 계층에서 읽는 오류를 발견했다. 실제 SW175 producer는 `admission_report.context_zero_init_sha256`에 기록하며 `initialization` 아래가 아니다. GPU 시작 전에 발견했으며 v1의 GPU·optimizer update는 0이었다. 원본 일곱 파일을 그대로 보존하고, 네 v2 파일에 올바른 계층 검증과 producer-shaped 회귀 검사를 추가했다. 검증을 생략하거나 모델 초기화를 변경하지 않았다. v1은 실패 증거 보존용이며 실행에는 위 v2 경로를 사용한다.

파일 SHA, 원본 CPU 거절 증거, 실제 모델 생성 CPU 결과, 시작 PID 관측은 `results_archive/started_source_preflight_20261016/`에 있다. 동료 브랜치 `patch_v2`도 이번 사이클에 fetch했고 `cab6c676e24a149a45ea960586c04e579f78d384` 이후 변경은 없었다. 데이터셋·checkpoint는 Git에 포함하지 않는다.
