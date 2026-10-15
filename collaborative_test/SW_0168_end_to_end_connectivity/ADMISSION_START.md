# SW0168 실제 spike 연결 학습: GPU 선행 검증 시작

2026-10-15. SW167의 frozen-source head 학습이 실패했으므로, 이번에는 별도 head 없이 실제 RGB encoder·graph·native core를 함께 학습한다. 정답 TRAIN mask를 사용하는 **supervised 실험**이며 SW165 최고 결과를 대체하거나 unsupervised 성과로 합치지 않는다.

Candidate는 물체 내부 연결의 bottleneck과 서로 다른 물체를 잇는 bridge에 pair-count 가중 credit을 준다. 비교군은 같은 4-neighbor edge에 균등한 local loss를 준다. 두 모델의 source·trainability·학습률·데이터·순서는 같고 loss만 다르다. Native positive-clamped spike Q를 그대로 사용한다.

서버에 연구 소스22개를 배포하기 전에 Python 구문과 CPU13개 테스트를 통과했다. 서버에서도 동일13개 테스트와 실제 TRAIN8장의 RGB/HDF mask/cache/source 일치 검증을 통과했다. 별도 독립 코드 검토에서 발견한 큐 timeout 분기 오류는 GPU 실행 전에 수정했고, 시간 초과 전에는 child를 중지하지 않는 동작을 검사했다.

최초 GPU 선행 검증 supervisor3667425/worker3667478은 B4 source/cache 검사를 통과한 뒤 B1 QCC 반환값을 Tensor로 취급한 API 오류로 종료됐다. 실제 production 함수는 NumPy 배열을 반환한다. 오류는 본 학습 전 발생했고 실패 상태·로그는 보존했다. 같은 시도를 재시작하거나 배포된 파일을 덮어쓰지 않았다.

`preflight_v2.py`는 NumPy 반환값을 직접 받도록 수정했다. 반환 API 회귀 검사를 포함한 CPU14개 테스트와 실제 TRAIN8 CPU admission이 서버에서 통과했다. 별도 `source_admission_v2_20261015` 큐 supervisor3667976/worker3668035가 GPU1에서 실행되는 것을 `/proc`와 NVIDIA 소유 PID로 확인했다. v2 fingerprint는 `2f6571e11e6d54ef884694e439ce29cf442c08a36dabfc846ac201f0b8e790b0`다. GPU0의 타 사용자 작업은 사용하지 않았다. **이 기록은 v2 실행 시작 증거이며 admission 통과나 성능 결과가 아니다.**

먼저 source B4 실제 six-trace/Q/A와 cache S/A를 검증하고, 별도로 새로 로드한 source의 production B1 QCC/SW164 partition·merge log를 각 arm 초기값과 비교한다. 이는 TRAIN8의 live source replay이며, 존재하지 않는 TRAIN B1 archive와 비교했다는 주장이 아니다. 각 arm 두 disposable B8 update(microbatch4)에서 실제 Q→component spike와 encoder/graph/native-core gradient 및 parameter 변화량을 검사한다. 본 학습 B16/64update와는 범위를 구분한다.

이 검증을 통과한 경우에만 고정512장·두 pass의 본 학습을 허용한다. Conditional frozen-support와 standalone learned-support 두 출력을 모두 평가한다. 목표는 고정 평가에서 성능이 실제로 개선되는지 확인하는 것이며, gradient 존재만으로 각 gating의 인과적 기여를 주장하지 않는다.

실행 증거: `results_archive/admission_started_20261015/server_records.json`, 최초 실패 `results_archive/admission_failed_20261015/server_records.json`, v2 `results_archive/admission_v2_started_20261015/server_records.json`. 생성 파일의 실제 저장 위치는 `collaborative_testing/generated/`이며 canonical 결과 경로는 junction으로 유지한다.
