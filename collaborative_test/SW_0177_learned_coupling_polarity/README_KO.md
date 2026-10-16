# SW177: 이미지별 연결 polarity 학습

Native SourceT의 실제 normalized projection으로 각 admitted edge의 multiplier를 학습한다. Signed와 동일 magnitude의 nonnegative control을 같은 초기 상태에서 비교한다. 두 arm 모두 원래 tail64 gradient, 동일 critical-component loss와 데이터를 사용한다. Forward SNN/gating 구조와 native32 mask 평가 계약은 유지한다. 자세한 등록 조건은 DESIGN_KO.md에 있다.

## 서버 등록 상태

로컬·서버 각각 17개 CPU 검사가 통과했다. 서버에서 실제 fresh SourceT/context 모델을 CPU에 로드했고, 초기 state SHA가 기존 대조군과 같은 f372b3a450ebeaf0d49e20d019d3c0676d4998cc1c49586e4c4e4c89e969d7f5임을 확인했다. 새 head는 33→32→1의 1121 parameters, final0 초기화이며 signed/control 초기 head state가 같다. Head 생성은 caller CPU RNG를 복원한다.

사전 테스트 큐 부모252661의 실제 명령행과 waiting_for_exclusive_gpu 상태를 확인했다. 등록 당시 모든 GPU에 외부 PID가 있어 큐가 기다린다. 외부 작업을 중지하거나 공유 GPU에 강제로 실행하지 않는다. 대기 중 CPU trainer 구현을 준비하며, 성공한 실제 GPU 사전 검사 receipt 없이는 학습을 시작하지 않는다.

다음 GPU 검사는 고정 TRAIN 첫 이미지68304 한 장에서 source와 zero-head 출력·두 mask 일치, 실제 component spike→head output gradient를 확인한다. 아직 실제 GPU 검사는 실행되지 않았고 optimizer update0이다. 이 CPU 등록 기록을 성능 개선이나 gating 필요성 증거로 취급하지 않는다. 이전 최고 모델은 유지한다.

원본 CPU 결과·초기 상태·큐 PID 증거는 results_archive/source_preflight_queued_20261016/에 보존했다. 결과 생성 파일은 collaborative_testing/generated 아래에 두고 source는 collaborative_test에 유지한다.

## 외부 GPU 진입 후 자동 재대기 큐로 복구

v1 부모252661은 장치를 확보한 뒤 외부 PID 진입을 감지해 자기 worker만 중지했다. 사전 검사 결과가 나오기 전의 자원 충돌이며 모델의 scientific rejection이 아니다. v1 원본 state와 빈 worker log를 그대로 보존했다.

원본 source/preflight 7개 파일은 유지하고 resource_queue_v2.py와 그 검사만 추가했다. 외부 owner 진입 시 자기 child만 중지하고 lease를 해제한 뒤 새 idle GPU를 기다린다. 각 시도의 로그·report는 별도로 보존한다. GPU 대기 시간은 worker 실행 제한시간에 포함하지 않으며 대기 자체에는 자동 종료시간을 두지 않는다.

새 큐 검사 3개가 로컬·서버에서 통과했다. 긴 GPU 대기, 외부 owner 진입, owned child만 중지, lease 해제·재확보 및 결과 보존을 한 경로로 검사했다. 실제 source fingerprint와 이전 CPU source 초기 상태 검증도 보존됐음을 서버에서 확인했다. 새 부모258907의 실제 명령행과 waiting_for_exclusive_gpu 상태를 확인했다. 아직 사전 GPU 결과·학습 업데이트는 없으며 기존 최고 모델은 유지한다. 이력은 results_archive/queue_recovered_v2_20261016/에 있다.

## 두 학습 큐 사전 등록

Signed 부모272532와 nonnegative 부모272541의 실제 명령행 및 waiting_for_v2_source_preflight 상태를 확인했다. 성공한 사전 검사 report와 source/code/큐 fingerprint가 확인돼야 CPU training admission이 성립하고 GPU lease를 요청한다. 사전 검사 또는 GPU 대기 시간은 worker 실행 제한시간에서 제외한다. 외부 GPU 진입 시 own worker만 중지하고 마지막 committed update에서 재개하며, GPU runtime binding이 있는 boundary는 같은 장치를 다시 확보한다.

새 학습 코드의 focused CPU 검사는 로컬·서버 각각 11개가 통과했고 로컬 combined suite는 31개가 통과했다. 서버에서 실제 fresh source를 두 번 CPU에 로드하여 signed/control combined initial state가 b2b59d00641935b9deb5ecd13200da020d096641acd46867f9108705144e805b로 같은지 확인했다. 실제 5개 optimizer family의 parameter 순서/설정도 같고 Adam state는 비어 있다. 실제 context 및 head 가중치를 변경한 후 boundary0에서 복원해 두 상태와 빈 Adam이 정확히 돌아오는지 검증했다. 이 CPU 점검을 GPU 사전 검사 성공 receipt로 대체하지 않았다.

계획은 seed1, TRAIN512×2, B16/micro4, 각64updates의 supervised paired pilot이다. 기존 native tail64를 쓰며 SW176 local-phase credit은 설치하지 않는다. 실제 head multiplier는 sparse A_graph 입력의 해당 forward에서 capture하고, floor가 포함된 dense A0로 head를 재계산하지 않는다. 음수 kernel 기여는 실제 Kuramoto의 네 항에서 측정하며 TRAIN8 진단의 수치는 settled512-1023/B4 batch aggregate로 기록한다. 최종 checkpoint tensor SHA와 Adam step summary는 저장된 실제 값으로 다시 검증한다.

아직 성공한 GPU 사전 검사 결과와 학습 업데이트·최종 평가 점수는 없다. GPU2 사전 시도 두 번은 외부 PID 진입으로 중지되어 재대기 중이며 부모258907은 정상이다. 최고 모델은 유지한다. CPU factory·모든 부모의 실행 증거는 results_archive/paired_training_queued_20261016/에 보존했다.
