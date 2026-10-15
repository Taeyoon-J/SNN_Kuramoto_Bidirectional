# SW0170 — actual-spike-derived causal synaptic current

2026-10-15, prospective registration/design. **아직 구현/배포/실행 없음. Supervised TRAIN-mask track.** SW169 final fixed320 실패를 보존하고 확장하지 않는다. Full320 live conditional FG .03837044, standalone0은 새 coding과 기존 affinity가 사용 가능한 수준으로 학습됐다는 근거가 아니다. Failed169 trained weights를 candidate initialization으로 쓰지 않는다. SW165 incumbent 및 .87/모든 gate/세 seed/native70k/scaling 목표는 유지한다.

## 단일 가설·source·신호 정의

Sparse event code가 기존 zero-lag centered-Pearson 네 component product에서 관계를 잃었을 가능성을 검정한다. 이 원인은 아직 확정되지 않았다. **동일 raw S에 causal synaptic state 하나를 더하는 것**만 새 비교다. Scalar alpha=exp(-1/16)는 고정 buffer이고 trainable/filter-width 선택은 없다.

```
Z_before_first_step = 0
Z_t = alpha*Z_(t-1) + (1-alpha)*S_t
```

S_t는 SW169 normalized hard-reset의 **actual emitted S=Eraw*g**다. Z는 continuous actual-spike-derived synaptic current이며 binary spikes라고 부르지 않는다. RGB/gamma/theta/직접 membrane 값은 Z/readout에 들어오지 않는다. Raw S가 기존 dendrite feedback에 들어가고 raw Ehistory가 binary hard reset을 제어하는 방정식은 그대로다. g=0이면 S=0이고 Z는 정상 decay한다; Z를 membrane처럼 hold하지 않는다.

Original SourceT strict-load/native32/T operator와 SW169의 고정 source TRAIN calibration r4 및 modified init(tau median e-fold16, threshold.5)을 재사용하되 immutable source/calibration/init hashes를 검증한다. Candidate는 fresh init이다. Kernel16은 앞선 prospective hypothesis에서 고정한 값으로 GT/활동률/VAL에서 고른 값이 아니다. 다른 width/threshold/clamp/time constant/LR/seed rescue는 금지한다.

## 비교와 시간·affinity 계약

* Candidate `causal_synapse`: 위 recurrence의 Z를 사용한다.
* Control `identity_synapse`: **Z=S 자체를 반환**한다. 동일 source/parameters/forward/데이터/loss/optimizer의 raw point-event affinity다.

양쪽 모두 full1024/settle512/live-tail64, B16 micro4다. Z는 시작부터1024 steps 전부 갱신하고960 boundary에서 값은 유지한 채 detach한다. 이후64 recurrence의 credit는 live다. Original hard-reset의 theta/history/dendrite/V/E/S boundary semantics는 변경하지 않는다. Initial identity/filter는 한 raw rollout을 공유할 수 있다. Candidate의 TRAIN Q_Z는 전체 settled512 Z에 production `spike_synchrony_affinity(Z.mean(D),Z,settle=0,affinity_mode='spike')`를 사용한다. Clamp/product/epsilon/.5 threshold는 그대로다.

Maximin loss와 energy 모두 같은 GPU Q_Z를 사용한다. Signed energy의 actual native A/gain 및 SW165 connected solver는 그대로다. Standalone support는 해당 Z로 native CPU QCC를 계산해 얻는다; CPU support Q와 GPU energy/loss Q를 같은 byte라고 가정하지 않는다. 두 outputs는 (a) frozen original SourceT F의 conditional diagnostic, (b) own Q_Z-QCC F의 standalone다. Conditional unseen predictor에는 frozen support source와 learned model의 두 rollout이 필요하고 standalone은 한 learned rollout이다. 양쪽 모두 모든 support node를 보존하고 GT count/background oracle를 쓰지 않는다.

## Stage1 — TRAIN8 rejection-only admission, 학습 fanout 이전

SW169와 같은 ordered512의 첫8 TRAIN IDs, B4 두 batches 및 source/r/F/feasible-GT-target lineage를 사용한다. TRAIN mask는 loss/아래 discrimination에만 사용하며 VAL/금지 reserve는 열지 않는다. **Optimizer0의 동일 actual S**를 identity/filter가 공유한다. Source/raw h·theta·g·E·S는 filter 때문에 변하지 않아야 한다.

1. Full1024 causal recurrence, zero init, tail boundary, identity S alias와 이전 raw Q/loss parity를 확인한다. Z와 Q_Z는 finite이며 temporal/node relation이 상수/allzero가 아니어야 한다. Self diagonal만으로 nondegenerate라고 판정하지 않는다. Raw E/activity와 Z activity는 별도로 보고한다.
2. 실제 `SW0168.connectivity_objective.constrained_maximin_loss`를 frozen target/F에 적용한 **8-image mean filtered loss < mean identity loss**를 요구한다. Empty-class 처리도 기존 API를 그대로 쓴다.
3. 같은 module의 `forest_routes`가 반환하는 positive_edges/negative_edges `(i,j,pair_mass)`로 각 image의 mu_pos=sum mass*Qij/positive_pair_mass, mu_neg=sum mass*Qij/negative_pair_mass를 정의한다. 양 represented classes가 존재하는 fixed-source eligible images에서 separation D=mu_pos-mu_neg의 image mean이 filter에서 identity보다 커야 한다. Eligible image set은 source target/F에만 의존한다. Eligible image가 없으면 중지한다. 모든 weighted class distributions/represented masses도 보고한다. 이것은 routed bottleneck/bridge 통계이지 모든 GT pair AUC나 FG-ARI가 아니다. 같은 edges를 강제로 공유하지 않고 각 affinity의 실제 deterministic forest를 사용한다.
4. Own filtered QCC의 **positive foreground가 있는 image가 최소4/8**이어야 한다. 이는 unseen candidate에 사전 고정한 noncollapse safety이며 untrained FG 점수 기준이 아니다. FG patch 수/group sizes는 모두 기록하지만 desired count/density는 강제하지 않는다.

1–4 통과 후에만 두 disposable logical updates를 수행한다. Production Q_Z→Z→actual S→encoder/graph/membrane 및 existing parameters까지 finite actionable gradients와 실제 optimizer changes를 확인한다. Direct Z-state gradient는 실제 recurrence states에서 확인한다. dL/dQ만 존재하고 positive-product clamp 뒤 dL/dS 또는 source family credit가 전부 죽으면 중지한다. Individual inactive parameter를 artificial update로 통과시키지 않는다. Theta/dendrite/core family names는 실제 `dendric_layer` 이름을 사용한다. Candidate/control trainability는 같고 추가 kernel parameter는 없다.

이 task-affinity admission을 통과하지 못하면 **64updates를 시작하지 않고 recipe를 닫는다**. Width, sample, threshold, normalization, loss 또는 cutoffs를 바꿔 다시 통과시키지 않는다. Admission은 supervised TRAIN feasibility일 뿐 segmentation improvement/gating usefulness 증명이 아니다. 이전 SW169 실패 기록은 별도 보존한다.

## Stage2 — 동일64updates 또는 입증된 identity 재사용

두 arm 모두 실제 RGB encoder+graph+native core+기존12 scalars를 학습한다. SW169와 동일 constrained maximin만 사용한다. Same ordered512×2passes=64 B16 updates/micro4, base/graph/native LR3e-5,integration LR1e-3, fresh Adam, clip1, init/order/RNG169001을 유지한다. Fresh candidate는 source init에서 시작한다. Warmup/추가 GT exposure/auxiliary loss/decoder는 없다. Actual runtime/peak memory를 두 disposable updates에서 측정하고 budget을 기록한다. Transactional logical checkpoints는 Z/boundary semantics 및 source/r/runtime fingerprint에 묶는다.

Completed SW169 `history_live`를 identity-control 결과로 재사용할 수 있는 경우는 원본 init/weights/raw forward/identity Q/loss/targets/F, B4 grouping/order/RNG,optimizer/projection/step count64,full1024/tail64/truncation이 **실제 source와 artifact에서 모두 검증된 경우뿐**이다. Identity branch가 S를 clone/변환하거나 loss arithmetic/parameter order/RNG를 바꾸어 이 동일성이 보장되지 않으면 fresh matched control을 수행한다. 원본 metadata를 새 experiment로 덮어쓰지 않고 cross-experiment adoption receipt로 보존한다. Reuse 여부를 score를 보고 선택하지 않는다.

## Fixed320 — transformation과 learning을 따로 판정

다음을 사전에 고정하고 전부 보고한다: optimizer0 filtered endpoint, optimizer0 identity endpoint(같은 init rollout 공유), final64 candidate, final64 identity. 각 모델/endpoint에서 conditional 및 standalone을 모두 생성한다. Initial-filter endpoint는 transformation의 효과를 측정하며 final learned candidate 개선으로 부르지 않는다. Final checkpoint 선택은 없다. 모든 예측/source references/ledger를 freeze·검증한 뒤 fixed IDs1320..1639/native32/modal4px의 기존 GT와 세 지표를 연다. GT count/allowed-label oracle/평가조건 변경은 없다. Reserve90640..90959는 읽지 않는다.

Final candidate의 **각 co-primary FG**가 SW164 seed1 .770932099와 final identity보다 높고 각각 paired95% CI lower>0이어야 한다. 같은 출력에서 initial-filter 대비 final learned FG도 높고 pairedCI lower>0이어야 학습에 의한 개선으로 판정한다. Conditional foregroundIoU per-image SourceT exact 및 objectIoU>=SW164, standalone 두 IoU means>=SW164도 필요하다. Initial-filter만 좋아지면 readout transformation signal로만 보고하며 이 학습 recipe를 통과로 바꾸지 않는다. Bootstrap10000/common320 image indices/RNG170. No seed/passes/width/loss sweep.

전체 gate 통과 시에만 동일 고정 recipe의 seed0/2를 별도 등록한다. 이후 actual E-forced1/history/reset interventions는 재롤아웃한 S→Z→mask를 비교해야 하며, Z 자체의 효용을 binary event 또는 전체 gate 필요성으로 치환하지 않는다. Fresh native32 70k와 positive unique-data scaling,unsupervised 성과와 .87 three-seed mean은 별도 검증이다. 생성 outputs는 중앙 `collaborative_testing/prepare_output.ps1` namespace와 immutable provenance archives를 사용한다.
