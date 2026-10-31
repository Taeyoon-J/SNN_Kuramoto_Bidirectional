# SW_0177_learned_coupling_polarity

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `293fa158f9dd165b6114a9acd8c54b5c673c8b9175c91665c2e7176396030f50`

````markdown
# SW177: 학습 가능한 coupling polarity

2026-10-16 prospective supervised paired pilot. 설계만 작성했으며 이 문서가 구현·배포·GPU 실행 증거는 아니다. SourceT→SW165의 정상적인 native32 spike coding에서 시작한다. 실패한 SW175/176 checkpoint와 Adam을 이어받거나 sealed source를 수정하지 않는다.

## 근거와 변경 범위

SW176 최종 seed1 conditional .716674/.614993/.597555, standalone .709090/.424322/.590484는 source SW164 .770932/.614993/.630523을 넘지 못했다. 독립 재계산 raw score SHA는 `7a4c4a3fcc0b00dd210703133bea6f687aef72b82fc3937cf5d4603eaaba4ab6`다. SW175 대비 standalone 개선이 있었어도 registered source gate는 실패했으므로 full-local phase credit 정책은 이번 모델에 가져오지 않는다.

이번 변화는 source graph의 각 admitted edge가 실제 phase interaction에서 가질 polarity를 이미지별로 학습하게 하는 것이다. Peer `cab6c676.../PV2_0092_signed_coupling/README.md`의 balance sweep은 pending이며 우리 계약의 성공 근거가 아니다. SW156의 학습 없는 uniform subtraction도 이미 실패했다. **Learned edge 선택**과 **sign 허용 대 magnitude-matched nonnegative control**을 시험하는 차이를 명시한다.

Native graph weight가 nonnegative라도 sine/phase-lag interaction은 순간적으로 양·음일 수 있다. 따라서 기존 dynamics에 모든 종류의 반발이 없었다고 주장하지 않는다. 새 sign은 그 native phase-lag kernels를 추가로 뒤집을 수 있는 기능이다. Foreground merge가 유일한 오류라는 가정도 하지 않는다.

## 공통 source와 head

양 arm은 fresh original SourceT seed1(native16 checkpoint를 native32로 spatial-map한 lineage)에 SW171의 같은 zero-final RF31 context를 붙인다. 원래 source CNN, preprocessing stats/clip, geodesic contrast는 frozen이다. Context/graph/native core 및 a_d,a_m,b의 3 tensors=12 scalar elements는 SW175와 동일하게 trainable이다. Native event threshold/surrogate, D/M/reset, delayed gate, gamma drive를 바꾸지 않는다.

Source graph의 **actual forward에서 한 번 계산된** normalized projection z[B,1024,16]을 capture한다. Recompute한 별도 teacher z를 쓰지 않는다. 현재 positive graph A_graph의 undirected support `i<j and A_graph[i,j]>0`에만 head를 적용한다. Support는 native directed top-k scatter 뒤 symmetrize한 실제 output에서 얻으며 학습된 edge 선택을 임의로 추가하지 않는다. Diagonal과 미채택 floor-only edge는 multiplier1이다.

고정 head 입력은 `[z_i+z_j, abs(z_i-z_j), grid_distance_ij/32]`의 33차원이다. Actual native32 graph의 32-grid physical distance를 사용한다. MLP33→32, SiLU, Linear32→1이며 final weight/bias는 모두0이다. Hidden layer는 CPU fork_rng 안 seed177001로 PyTorch 기본 초기화하고 caller RNG를 복원한다. Context/model/shuffle seed171001은 유지한다. 각 undirected pair는 한 번만 계산하여 mirror하므로 multiplier가 정확히 대칭이다.

Head는 coupling을 생성하는 module이며 mask classifier가 아니다. 추론 mask는 실제 emitted S에서 production Q/QCC 및 SW165 partition으로만 나온다. RGB/GT/scene object count/직접 z mask를 classifier에 전달하지 않는다. Head는 checkpoint에 포함되는 registered module로 두고 optimizer의 별도 family에 넣으며 graph family와 parameter 중복을 금지한다. Source strict-load 뒤에 head를 붙인다.

## 두 arm과 실제 prepare 경로

Head output r에 대해 `m=1-2*tanh(r)`를 고정한다.

- `signed`: q=m.
- `nonnegative`: q=abs(m).

초기 r=0, q=1이므로 두 arm의 source forward와 초기 gradient가 같다. 같은 r에서 magnitude는 같고 negative sign 허용 여부가 다르다. 학습 후 parameter trajectories와 magnitudes는 달라질 수 있으므로 최종 차이를 완벽한 sign-only 분해라고 주장하지 않는다. m=0의 abs subgradient는 PyTorch 기본0이다. Temperature/smoothing/threshold를 추가하지 않는다.

원래 positive A_graph를 원래 `Kuramoto.prepare_coupling`에 먼저 넣어 native `(A0,Lcos0,Lsin0)`를 얻는다. Native ReLU/floor, cost normalization, learned direction/alpha 계산을 모두 유지한다. 그 다음 tuple을 `(A0,Lcos0*q,Lsin0*q)`로 바꾼다. 음수 adjacency를 native prepare에 직접 넣으면 ReLU가 제거하므로 금지한다. Division으로 alpha를 재구성하거나 signed graph로 lag를 다시 만들지 않는다.

Native factorized forward의 원래 네 bmm interaction 항이 수정 kernels를 실제 사용해야 한다. A0는 positive structural graph 및 원래 prepare tuple[0]로 남긴다. 실제 signed interaction weights B=A0*q는 별도 capture/hash/diagnostic으로 구분한다. q=1일 때 곱셈 경로가 source와 exact parity를 만족해야 하며, zero head라는 이유로 gradient가 필요한 경로를 return-native로 short-circuit하지 않는다.

SW156의 instance-scoped prepare wrapper/known-tensor tests는 경로 참고로 재사용할 수 있으나 beta/U subtraction 수식은 재사용하지 않는다. Shared class/global functions를 영구 monkeypatch하지 않고 정상/예외 시 복구한다. Static graph/no feedback/no spike-pulse source 제약도 유지한다.

## 고정 학습 계약

Credit 정책은 **원래 SW175 native tail64**다: total1024, settle512, no-grad prefix960, boundary state detach, tail64 live. SW176 theta pre-hook/full-local policy를 설치하지 않는다. 이 선택은 승리한 정책이라는 뜻이 아니라 실패한 policy를 새 architecture에 섞지 않기 위한 기준이다.

공통 loss는 frozen SW175 `critical_component_objective.py`다. Positive GT-connected maximin forest의 pair-mass weighted `amax_d((1-rho_d)^2)`와 원래 negative Q²를 사용한다. Production Q는 실제 settled component S의 positive-clamped Pearson product 그대로다. 새로운 RGB/auxiliary/teacher/lambda 항은 없다.

같은 admitted ordered512 TRAIN RGB/mask cache를 쓴다. NumPy RandomState171001의 연속 두512 permutation, B16/micro4,64 logical updates/1024 optimizer exposures로 고정한다. 각 image는 두 번씩이다. Source/context/기존 optimizer family names/order와 per-update CPU/CUDA seed 규칙은 SW175를 유지한다. Fresh Adam LR는 context3e-4, graph/native-core3e-5, integration1e-3, 새 head1e-3이다. Global clip1, update당 Adam1회와 integration projection1회다. Head 때문에 SW175 완료 arm은 matched control을 대체할 수 없으며 두 arm을 모두 fresh로 학습한다.

두 arm의 initial model/optimizer configuration·source/cache/globalID order·head state SHA가 같음을 bind한다. Initial optimizer state는 fresh이며 서로의 trained model을 load하지 않는다. Boundary마다 model/head/Adam/RNG/history/cursor/runtime 및 code/design hashes를 보존한다. Retry는 마지막 completed logical update만 복구하고 partial microbatch gradient를 이어받지 않는다.

## 최소 preflight와 관측

CPU checks: q=1 source tuple/interaction equality; known negative multiplier가 native ReLU 이후 네 interaction 항에 남음; signed/abs magnitude 및 zero-head forward/초기 gradient parity; symmetric support/diagonal/floor 유지; scoped 복원. Bool receipts는 Python bool로 serialize하고 grad element finite와 float64-accumulated norm을 구분한다.

GPU no-update preflight는 같은 cache의 **TRAIN position0 한 장**(실제 globalID를 manifest에서 bind; 기존 cache에서는68304)만 쓴다. Source와 두 zero-head arm의 같은 RNG/B1 gamma/six traces/positive A0/projection/production Q 및 conditional/standalone masks exact parity를 확인한다. 실제 critical loss→emitted S→head final layer gradient가 finite-positive여야 한다. Hidden gradient0은 zero-final 초기화의 예상 결과다. CPU loss fixture나 sibling membrane trace를 actual emitted S credit의 대체로 삼지 않는다. Source/parameters/RNG 변경 없이 종료하고 VAL/reserve GT는 읽지 않는다.

첫 두 실제 committed updates에서 finite loss/parameter gradients, output-layer 첫 credit/둘째 hidden credit·parameter change 및 peak memory/time을 기록한다. 별도 disposable optimizer exposure는 추가하지 않는다. Fixed TRAIN8 positions256..263은 initial/final 관측만 하며 loss 감소를64-update 진행 gate나 checkpoint 선택에 쓰지 않는다. 비유한 원소/실제 경로 dead/forward·provenance 불일치만 진입 실패로 보존한다. 각 scalar gradient가0이라는 이유만으로 새 threshold를 만들지 않는다.

Initial negative edge0은 정상이다. 학습 중/마지막 TRAIN8에서는 negative edge fraction, A0-weighted negative mass, B amplitude와 **negative kernels만 mask한 실제 coupling-term norm**을 기록한다. 원래 네 bmm 항으로 같은 trajectory의 additive contribution을 분해하며, 그 값이 counterfactual 전체 rollout 효과라고 주장하지 않는다. 64회 뒤에도 실제 negative interaction이 없으면 repulsion 메커니즘을 검증했다고 말하지 않는다. Bias/scale/LR/passes를 사후 조정해 sign을 강제로 켜지 않는다.

## 고정320 task gate와 종료

추론은 original native32 input/1024steps/settle512, actual S의 production Q, positive structural A0와 실제 trained graph gain, 원래 SW165 induced4-neighbor signed-energy greedy solver다. B를 positive structural readout A로 오인하여 넣지 않는다. Slot/metric/GT modalization/background/min-size 계약을 변경하지 않는다.

두 모델 각각 (1) frozen SourceT foreground support를 쓰는 conditional attribution output, (2) 같은 learned rollout의 native CPU-QCC support를 쓰는 standalone practical output을 모두 제출한다. Conditional만 source foreground equality를 요구하며 standalone을 그 equality로 검증하지 않는다. Conditional의 추가 SourceT support compute/immutable artifact provenance는 따로 명시한다. 모든 final320 predictions/IDs/protocol hashes를 freeze한 뒤 GT를 읽는다.

고정 baseline은 SW164 seed1 .7709320986/.6149934303/.6305234820이다. QCC-only SourceT .755273과 혼동하지 않는다. Signed candidate가 conditional와 standalone **각각** baseline 및 새 nonnegative control보다 FG mean이 높고 common-image paired10,000 bootstrap(seed177)의95% lower delta CI가 모두>0이어야 한다. 두 IoU means도 각각 baseline/control 이상이어야 한다. Negative sign 사용 여부와 모든 실패 arm을 함께 보고하며 두 output 중 좋은 것만 선택하지 않는다.

Task gate 실패면 recipe를 닫고 seed0/2/70k/계수 sweep을 하지 않는다. Sign이 사용되지 않았는데 성능이 오르면 positive reweighting으로만 해석하며 repulsion/all-gate 증거로 승격하지 않는다. 통과한 경우에만 동일 recipe seed0/2를 등록하고 all-three 판단 후 fresh native32/70k와 fixed-update/fixed-pass unique-data scaling을 분리해 검증한다. 이번 실험은 TRAIN masks를 쓰는 supervised track이며 unsupervised .87/모든 gate 필요성/70k superiority를 이미 달성했다는 주장이 아니다.

````

</details>

## README_KO.md

<details>
<summary>README_KO.md</summary>

Source SHA256: `d9ac3fd7687b6b93661a5a4924ae35b50d99fb436f53962ac2ea72cc1a923db7`

````markdown
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

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `cdd71a302c48979adb2cd47e91f005d1445024176f66e321837004323358a8e6`

````markdown
# SW177 최종 결과: 기존 모델 개선 실패

고정 native32 평가320장, seed1의 supervised TRAIN512×2 passes/64 updates 비교다. 기존 최고 세 seed 평균이나 새70,000장 학습 결과가 아니다. 두 모델의 학습 checkpoint·Adam·순서와 불변모델 예측 receipt를 검증하고, 양쪽320 예측을 확정한 뒤 GT를 열었다. Production 재평가와 별도 metric/평균/bootstrap/gate 재계산이 일치했다.

| 모델 / 출력 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 SW164 seed1 | 0.770932 | 0.614993 | 0.630523 |
| Signed / 원본 foreground 유지 | 0.725166 | 0.614993 | 0.613633 |
| Nonnegative / 원본 foreground 유지 | 0.693004 | 0.614993 | 0.602618 |
| Signed / 자체 native-QCC foreground | 0.718381 | 0.388679 | 0.600379 |
| Nonnegative / 자체 native-QCC foreground | 0.680424 | 0.471769 | 0.590524 |

Signed의 기존 모델 대비 FG 차이는 원본 foreground 유지 시 −0.045766, paired95% CI [−0.054653,−0.037457], 자체 foreground에서는 −0.052551, CI [−0.063636,−0.041754]다. Bootstrap은 같은320 이미지·10,000 resamples·seed177이다.

Signed는 새 nonnegative control보다 FG가 높았다(conditional +0.032162, standalone +0.037957, 두 CI 하한>0). 그러나 기존 모델보다 낮았고 자체 foreground IoU는 control보다도 낮았다. 등록된 종합 조건을 통과하지 못했으므로 최고 모델로 승격하지 않고 seed0/2·70k 확장도 하지 않는다.

## 학습 신호와 성능을 구분

실제 GPU 사전 검증은 원본과 zero-head signed의 forward/mask exact parity, 실제 emitted S→head gradient를 통과했다. 두 실제 첫 committed updates에서도 spike/head credit과 parameter 변경을 확인했다. 이러한 학습 작동 증거가 최종 mask 개선을 보장하지는 않았다.

TRAIN8 loss는 초기0.3869964841에서 signed0.3674587768, nonnegative0.3204177671로 내려갔다. Loss가 더 낮았던 control의 FG가 더 나빴다. 현재 절대 pair/forest 목적함수와 최종 mask 구분 사이의 불일치가 다음 검증 가설이며, 이번 결과만으로 정확한 원인을 모두 확정하지 않는다.

Signed의 최종 TRAIN8에서 negative edge fraction과 실제 negative-kernel contribution RMS는 모두0이었다. 이것은8장 관측으로, 전체 TRAIN/VAL의 음수 사용 부재를 증명하지 않는다. Signed/control 성능 차이를 repulsion 또는 모든 gating의 기여로 해석하지 않는다. 실패 후 bias/LR/온도/passes를 바꿔 이 recipe를 연장하지 않는다.

다음 후보는 실제 spike temporal code의 상대적인 object-class competition이다. 전체 TRAIN modal patches와 BG를 loss에서 사용하고, signed-rho 평균 CE와 production-Q CE를 비교한다. 새 head/decoder를 붙이지 않으며 추론과 평가 계약을 유지한다. 각 arm의 모델 유용성과 mean-rho 가설의 상대적 우위는 별도 판정한다. 이는 별도 supervised 실험으로 등록하며, 이번 실패 결과를 대체하지 않는다.

## 원본 및 검증 근거

`results_archive/completed_score_20261016/`에 원본 evaluation, 양쪽 protocol/queue/불변모델 receipt, 독립 검증 및 검증 코드를 보존했다. Score byte SHA는 `860a5d893b2b6e098cf7d5ebaf5718ee124a396319ea357dbad9d77e9c19e3eb`다. 모델/데이터/평가 변경 없이 고친 초기 queue 오류도 `evaluation_recovered_20261016/`에 별도 보존했다. 이번 cycle의 동료 브랜치 fetch는 기존 `cab6c676e24a149a45ea960586c04e579f78d384`와 같았으며 새 성공 결과로 취급하지 않았다.

````

</details>

## VERIFICATION_SCOPE_ADDENDUM_KO.md

<details>
<summary>VERIFICATION_SCOPE_ADDENDUM_KO.md</summary>

Source SHA256: `ff72c181b1dd904e35ac60e84e98bb6a09e5cbdabe76a4fe4e73757411da4837`

````markdown
# SW177 초기 parity 검증 범위 정정

2026-10-16. GPU preflight 합격 결과와 학습 업데이트를 관측하기 전에 기록한다. Frozen DESIGN_KO.md와 실행 파일, 실행 중 큐는 수정하지 않는다.

DESIGN_KO.md의 “Source와 두 zero-head arm” GPU 직접 비교라는 문구는 현재 구현보다 넓다. 실제 preflight.py의 직접 GPU 비교 대상은 **SourceT와 signed zero-head arm**이다. 독립 nonnegative GPU rollout은 실행하지 않았다. 이 문서는 이를 두 arm의 GPU 실측으로 보고하지 않도록 검증 범위를 정정한다.

Nonnegative의 초기 동등성은 다음 근거로만 판정한다.

- 두 실제 CPU factory의 초기 source/head state가 같고 final head weight/bias가 0이다. 서버 CPU 검증에서 combined initial state SHA는 두 arm 모두 `b2b59d00641935b9deb5ecd13200da020d096641acd46867f9108705144e805b`, head SHA는 `35cb30c146b60260e7f8e66b2cde86829effe8e64af62cc82b90429411363bab`였다.
- 같은 유한 입력에서 초기 r=0, m=1−2tanh(r)=1이다. 두 mode의 차이는 m에 대한 abs뿐이므로 abs(1)=1이고 해당 지점의 도함수도 1이다. CPU operator 검사는 두 mode의 native tuple, 초기 forward 및 gradient 동등성을 검증한다.
- **실제 signed GPU source-parity preflight가 합격해야** 이 근거를 GPU 학습 진입에 적용한다. CPU 검사나 대수적 동일성으로 GPU 합격 receipt를 대체하지 않는다. 합격 receipt가 나오면 원본 report 경로와 byte SHA를 별도 결과 기록에 연결한다.

이 논증은 초기 zero-final 상태에 한정한다. 학습 후 signed/nonnegative 궤적, magnitude 또는 mask의 동등성을 의미하지 않는다. 각 arm은 별도 fresh optimizer로 학습하며 각각의 실제 산출물을 평가한다.

## 고정 근거 파일

| 파일 | SHA256 |
|---|---|
| DESIGN_KO.md | 293fa158f9dd165b6114a9acd8c54b5c673c8b9175c91665c2e7176396030f50 |
| polarity_operator.py | f85de0ae29fe5f71bce67cffc1f4865de565a417b5508c013fc2ed8f04ae6099 |
| test_polarity_operator.py | ceea3ca0a5724fbb3d1288ca38cd162d32368f9bf0469854e2361f8c9a3a401f |
| preflight.py | 31160b3accd94f72ccc1486b248c2da199fd71bb456852343b4d9113c19160e8 |
| train.py | d9c1e3d77a6218b6663954362deebd947ce1ef6095beb1b26d50c119091777a4 |
| test_train.py | 3e9554ec83e55254322fad8e2c970dac6722f98eb5cd39d39ea0be586d0b83bf |

학습 데이터·순서·64 updates·loss·두 평가 endpoint·baseline/control 비교·bootstrap 및 IoU 통과 조건은 DESIGN_KO.md 그대로다. 이 정정은 성능 결과나 repulsion/gating 기여의 증거가 아니다.

````

</details>
