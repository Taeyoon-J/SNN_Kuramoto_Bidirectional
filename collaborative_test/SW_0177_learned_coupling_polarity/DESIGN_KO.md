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
