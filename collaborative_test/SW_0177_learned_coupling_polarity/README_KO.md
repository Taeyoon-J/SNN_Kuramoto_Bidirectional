# SW177: 이미지별 연결 polarity 학습

## 최종 판정 (2026-10-16)

**완료: 기존 모델보다 FG와 Object IoU가 낮아 recipe를 닫았다.** Seed1 signed conditional은0.725166/0.614993/0.613633, standalone은0.718381/0.388679/0.600379였다. 기존 SW164 seed1은0.770932/0.614993/0.630523이다. 최고 모델과 세 seed 평균은 갱신하지 않으며 seed0/2·70k로 확대하지 않는다. 최종 점수·해석·bootstrap 및 SHA 근거는 [RESULT_KO.md](RESULT_KO.md)에 있다. 아래 학습/예측/점수 대기 문구는 당시 실행 이력이다.

## 두 학습 완료·fixed320 예측 시작 (2026-10-16)

두 arm 모두 등록된64 updates/1024 exposures를 완료했고 checkpoint tensor SHA·Adam·학습 순서·terminal manifest receipt 검증을 통과했다. Signed manifest SHA는 `547beef232852ea7b52c333e42b263b0775a509230a48d587b56c713c3cc4aee`, nonnegative는 `d7cc112d7b538fa23d51a2cdf854ae12fe4f48c869d9639166f2f817d5d63a43`이다. 실제 완료 checkpoint의 CPU strict-load도 prediction 큐의 GPU 확보 이전 절차를 통과하여 signed GPU2 worker332139가 시작됐다. Nonnegative 예측과 GT-last score 큐는 기존 등록을 유지한다.

고정 TRAIN8 진단 loss는 초기0.3869964841에서 signed0.3674587768, nonnegative0.3204177671로 내려갔다. **Signed의 최종 TRAIN8 전부에서 실제 negative edge fraction과 negative-kernel contribution RMS는0**이었다. 이는8장 관측이며 전체 이미지에 대한 단정은 아니지만, 현재 repulsion 기여를 입증하지 못했다. 최종 FG-ARI/IoU 개선도 아직 미확인이다. 판정 조건이나 LR/bias/passes를 사후 변경하지 않는다. 검증 기록은 `results_archive/training_complete_20261016/`에 보존했고 기존 최고 모델은 유지한다.

## GPU 재확보 대기 판정 복구 (2026-10-16)

Nonnegative 비교 모델은64 updates를 완료했으며 실제 checkpoint tensor SHA·optimizer·학습 순서·terminal manifest receipt를 독립 검증했다(manifest SHA `d7cc112d7b538fa23d51a2cdf854ae12fe4f48c869d9639166f2f817d5d63a43`). Signed 모델은 기존 boundary에서 GPU2 학습을 재개했다. 아직 두 모델의 최종 task 점수는 없다.

Signed 부모가 정상적인 `waiting_for_stable_gpu`로 전환할 때 첫 평가 큐의 consumer active-state 목록에 해당 상태가 빠져 prediction 두 부모와 score 부모가 종료됐다. 앞선 검사는 consumer 자신의 enum을 순회했기 때문에 producer 상태 누락을 잡지 못했다. 오류는 예측·VAL GT 접근 전에 발생했다. 실제 종료 PID 부재를 확인하고 원래 state/log 및 초기 source revision을 보존한 뒤, active-state 항목1개와 독립적인 producer 다섯 상태·정확한 matching-arm 부모를 사용하는 회귀 검사를 추가했다. 모델·loss·학습 checkpoint는 변경하지 않았다.

수정 후 로컬·서버 focused CPU 검사15개가 통과했다. 예측 부모323207/323225와 score 부모323245의 실제 명령줄·대기 상태를 확인했다. 실패한 runtime은 서버의 기존 폴더명 뒤 `_failed_stable_wait`를 붙여 보존했고, 복구 기록은 `results_archive/evaluation_recovered_20261016/`에 있다. 아래316052/316067/316085 등록 기록은 복구 이전 이력이다. 기존 학습을 처음부터 다시 실행하지 않았다.

## 학습 후 자동 평가 큐 등록 (2026-10-16)

두 prediction 부모316052/316067과 score 부모316085의 실제 명령줄 및 대기 상태를 확인했다. 두64-update 학습 산출물의 manifest/실제 tensor SHA/optimizer와 각 queue receipt를 검증한 뒤 같은320 RGB(IDs1320–1639)로 conditional 및 standalone 출력을 만든다. Conditional만 SourceT foreground를 유지하며 standalone은 해당 모델의 실제 native-QCC foreground를 유지한다. Signed/nonnegative 모든 예측과 protocol/row ledger를 확정한 후에만 VAL GT를 읽어 세 지표와 등록된 paired bootstrap을 계산한다. Prediction GPU도 exclusive idle 장치만 사용하고 외부 PID가 진입하면 own worker만 양보한다.

평가 관련 focused CPU tests는 로컬·서버 각각14개가 통과했다. 서버에서는 실제 SourceT factory와 실제 polarity head를 생성하고, 학습 없이 의도적으로 변경한 head를 임시 checkpoint에 저장한 뒤 production loader로 strict 복원하여 tensor SHA/가중치 일치를 확인했다. 이 CPU 검사는 완료된 학습 모델의 평가나 GPU prediction admission이 아니다. 실제 완료 checkpoint의 CPU strict-load는 큐에서 GPU 확보 전에 다시 수행한다. 검증·배포 파일 SHA·부모 실행 근거는 `results_archive/evaluation_queued_20261016/`에 있다.

실제 첫 두 committed 학습 update의 checkpoint envelope/payload 무결성·학습 순서·spike→head credit도 두 arm에서 확인했다. 둘째 update에서 hidden head gradient가 양수였고5개 optimizer family의 가중치가 실제 바뀌었다. 해당 읽기 전용 관측은 `results_archive/first_committed_credit_20261016/audit.json`에 보존한다. 학습 작동 증거이며 성능 향상의 증거는 아니다. 최종 fixed320 점수는 아직 없다.

## 실제 GPU preflight 통과 (2026-10-16)

GPU2 attempt04에서 SourceT와 signed zero-head의 gamma, 여섯 traces, positive A0, projection, production Q, conditional/standalone mask가 모두 exact parity를 통과했다. 실제 emitted S gradient L2는 0.005588407173510742, head final-layer combined gradient L2는 0.01953709491430246이며 모두 유한하다. Zero-final 초기 hidden gradient0은 예상한 값이다. Optimizer update0, VAL GT 미사용, source state 변경 없음이다. 이 결과는 성능 향상이나 학습 후 음수 연결 기여의 증거가 아니다.

원본 report byte SHA는 `be9f9c5e62efb3f87b2f43a4372f3842adf72f020d6d8d8d8772f0dc71810c38`이다. Queue/code/source fingerprint를 실제 서버 모듈로 다시 검증했고, 초기 model SHA도 기존 CPU factory와 일치했다. 원본 report/state와 검증 receipt는 `results_archive/source_preflight_completed_20261016/`에 보존했다. 독립 nonnegative GPU rollout은 수행하지 않았으며 초기 동등성의 정확한 검증 범위는 `VERIFICATION_SCOPE_ADDENDUM_KO.md`에 따른다.

등록된 signed 학습이 GPU2에서 이어 시작됐고 nonnegative 학습은 exclusive GPU를 기다린다. 아래 등록 이력의 “GPU 결과 없음”은 등록 당시 상태이며 이 절의 사전 테스트 통과로 갱신된다. 아직 최종 학습·fixed320 성능 점수는 없다.

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
