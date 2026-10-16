# SW176 local phase gradient: 시작 기록

긴 Kuramoto 재귀를 통한 학습 신호 전달만 바꾸는 실험이다. 자세한 변경·대조·확대 조건은 DESIGN_KO.md에 사전 기록했다. 모델의 forward 연산, native component spike, Q/A, threshold와 mask readout은 바꾸지 않는다. D/M/event/reset의 장기 gradient는 여전히 남아 있다.

현재 실행 경로는 `python -m collaborative_test.SW_0176_local_phase_credit.resource_queue_v3`이다. 이 단계는 TRAIN 위치0의 이미지 한 장에서 no-update source 검증 및 full1024 gradient를 측정하는 실험이다. 64-update 학습이나 검증 성능 결과가 아니다.

로컬·서버 각 15개 CPU 검사와 실제 TRAIN 캐시 준비, fresh source+context 모델 생성을 통과했다. 생성한 초기 state SHA `f372b3a450ebeaf0d49e20d019d3c0676d4998cc1c49586e4c4e4c89e969d7f5`가 완료 SW175 critical-component 대조군의 초기 state SHA와 정확히 일치했다. 이후 GPU1 worker의 실제 PID를 확인했다. 결과 대기 중이며 기존 최고 모델은 유지한다.

최초 v1의 실제 CPU 모델 생성 검증은 context 초기화 SHA를 잘못된 메타데이터 계층에서 읽는 오류를 발견했다. 실제 SW175 producer는 `admission_report.context_zero_init_sha256`에 기록하며 `initialization` 아래가 아니다. GPU 시작 전에 발견했으며 v1의 GPU·optimizer update는 0이었다. 원본 일곱 파일을 그대로 보존하고, 네 v2 파일에 올바른 계층 검증과 producer-shaped 회귀 검사를 추가했다. 검증을 생략하거나 모델 초기화를 변경하지 않았다. v1은 실패 증거 보존용이며 실행에는 위 v2 경로를 사용한다.

파일 SHA, 원본 CPU 거절 증거, 실제 모델 생성 CPU 결과, 시작 PID 관측은 `results_archive/started_source_preflight_20261016/`에 있다. 동료 브랜치 `patch_v2`도 이번 사이클에 fetch했고 `cab6c676e24a149a45ea960586c04e579f78d384` 이후 변경은 없었다. 데이터셋·checkpoint는 Git에 포함하지 않는다.

## v2 시작 후 확인된 초기화 오류

v2 부모/worker가 종료됐고 실제 source forward 이전 `reset_peak_memory_stats`에서 `Invalid device argument`가 발생했다. CPU 준비와 fresh 초기 state 검증은 통과했지만, 메모리 계측 API 호출 전에 CUDA device를 명시적으로 초기화해야 했다. optimizer update와 성능 결과는 없다. 원본 v2 코드·로그는 `results_archive/terminal_gpu_initialization_failure_20261016/`에 보존했다. CPU admission guard 뒤에서 device를 초기화하도록 새 버전을 검증하며, 이 기록을 모델 가설의 실패로 해석하지 않는다.

## CUDA 초기화 v3 복구 시작

CPU 준비와 uninitialized-CUDA guard를 통과한 뒤 `torch.cuda.set_device`를 호출하고, 그 다음에 peak memory를 초기화하도록 수정했다. public run_preflight 경로의 호출 순서를 검증하는 회귀 검사를 추가했다. 로컬·서버 각 16개 검사와 실제 fresh 모델·초기 state 동일성을 다시 통과했다. 새 v3 source와 별도 runtime을 사용하며 기존 v2 오류 로그를 보존했다. GPU1의 실제 부모202412·worker203150 실행을 확인했다. 결과는 아직 대기 중이며 초기화 수정이 모델 성능 개선의 증거는 아니다. 증거는 `results_archive/started_cuda_recovery_v3_20261016/`에 있다.

## v3 검증기 실패: 과학적 음성 판정 아님

v3는 forward 동일성·Q와 emitted spike gradient 검사를 지나 family 검증에서 종료됐다. 그러나 실제 `_grad_stats` 생산자는 `numpy.bool_(True)`를 반환하고 validator는 Python `True`와의 identity를 요구해 정상 값도 거절한다. 실제 유한한 parameter gradient를 producer에 넣어 validator가 잘못 거절하는 것을 CPU에서 재현했다. 따라서 현재 로그로 모델 gradient가 비유한하다거나 학습 정책이 실패했다고 판단할 수 없다. parameter gradient의 실제 수치는 보고서가 쓰이기 전 종료돼 미확정이다. 원소 발산과 float32 norm overflow도 구분되지 않았다.

가설 변경 SW177은 보류한다. 검증기 타입을 수정하고 실제 producer→validator 회귀를 추가한 뒤, 동일176 정책의 bounded 실험으로 수치를 확보한다. 원본 실패 로그와 CPU 증거는 `results_archive/invalid_family_guard_v3_20261016/`에 보존했다. optimizer update는 0이며 최고 모델은 유지한다.
