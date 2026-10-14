# SW0163_relation_context_assignment: frozen sourceT + supervised relation-context assignment

2026-10-14 prospective design. 사용자가 승인한 **fixed-core-first** 실험이다. 구현·서버·queue 작업은 하지 않는다. SW160/161/162 및 shared source 파일을 변경하지 않는다. MST loss 제안으로 이 실험을 대체하지 않는다.

## 질문과 네 대조군

SW161의 raw-spike binder는 current QCC보다 나빴다. 이번에는 sourceT의 graph와 relative spike 관계를 별도 message로 제공하여, raw temporal projection에 빠진 relational structure가 실제 assignment 학습을 돕는지 검사한다. TRAIN mask supervision을 쓰므로 unsupervised leaderboard와 분리한다.

모든 arm은 같은 source97 seed1→native32 sourceT,동일 actual spikes,동일 head 파라미터 shape/초기값,동일 mask loss를 사용한다. Source encoder/patcher/graph/core/integration는 모두 frozen이다. SourceT graph power2/effective contrast8은 원래 SW143 operator다. SW159 threshold 및 SW158 emitter는 설치하지 않는다.

|arm|Q message|A message|
|---|---|---|
|raw_only|0|0|
|Q_only|live input|0|
|A_only|0|live input|
|Q_A|live input|live input|

여기서 모든 arm의 raw input은 actual spikes다. **A_only는 raw spikes를 제거한 A-only model이 아니다.** QA-vs-A_only는 Q-context의 추가 기여만 검사하며 spike/gate 자체의 필요성을 증명하지 않는다. Inactive channels는 같은 shape의 zero tensor로 대체한다. 파라미터를 없애거나 arm별 capacity를 바꾸지 않는다.

## 정확한 head

SW135 `NativeSpikeSlotBinder`의 source component input `[B,4,1024,512]`→2048-dimension patch vector→Linear64→LayerNorm을 그대로 사용해 `h0[B,1024,64]`를 만든다. Head base seed는161001로 고정하며 failed SW161 trained weights는 재사용하지 않는다. 새로운 message module만 별도 RNG163001에서 초기화하여 base initialization을 교란하지 않는다.

`Q0=Q*(1-I)`, `A0=A*(1-I)`.

`RQ=Q0/Q0.sum(-1,keepdim=True).clamp_min(1e-8)`; `RA`도 같은 rule. Zero-sum row의 message는0이며 self loop를 보충하지 않는다.

두 번 반복하여 `h_next=h+F(concat(h,cQ*(RQ@h),cA*(RA@h)))`를 적용한다. cQ/cA는 위 arm의 고정0/1이다. F는 **shared** Linear192→128,ReLU,Linear128→64이며 두 update가 같은 F를 쓴다. Final Linear weight/bias는 둘 다0으로 초기화한다. 별도의 post-residual normalization으로 zero-init parity를 깨지 않는다.

Context h를 기존 shared key/value 및64-dimension slot query/GRU/slotMLP의3-iteration binding에 전달한다.11 exchangeable slots,동일 fixed initial noise,softmax P가 최종 assignment다. Q/A를 flatten해서 별도 learned projection에 넣거나 RGB/γ/θ/XY를 추가하지 않는다. Actual A는 source image context를 포함한다는 점은 명시한다. A context의 효과를 모든 gate의 necessity라고 부르지 않는다.

Final message projection이0이므로 초기 네 head의 P가 fresh SW161 raw head와 같아야 한다. First update에는 message hidden/input weights의 gradient가0일 수 있다. Second update에서 활성 channel의 실제 credit을 검사한다. Raw/Q/A controls의 inactive columns는 gradient0이어도 정상이다.

## 실제 cache/API와 context 생성

기존 SW161 source cache를 **read-only** 재사용한다. Path는 `SW_0161_supervised_spike_partition/cache.py::CACHE_ROOT`; manifest SHA는 `b4125a81012bf46cf38ff512bf251452f511524c42a2d2364c34783de04302b3`다. `train.validate_cache`/`cache.validate_final_cache`의 실제 source bindings/row ledger validation을 수행하고 `_AtomicTraceCache.load_row(index)`가 반환하는 float32 `[4,1024,512]` 및 modal32 TRAIN targets를 읽는다.512 source IDs/order/RGB hashes를 바꾸거나 float16 trace를 float32로 cast해 대체하지 않는다.

Q는 saved actual settled spikes에서 production `spike_synchrony_affinity(activity=S.mean(1),components=S,settle=0,affinity_mode='spike')`로 만든다. 이미512 settled frames이므로 settle512를 다시 적용하지 않는다. Context cache Q는 원래 B4 grouping에서 CUDA float32 production helper로 계산·고정한다. Inference도 같은 CUDA float32 production function을 B1으로 사용한다. B4와 B1의 수치가 bit-exact하다고 주장하지 않는다. Native sourceQCC reference는 원래 `_qcc_labels` 경로를 유지한다.

A는 **512번 full rollout을 다시 하지 않고** source RGB/weights로 static setup만 계산한다. SW155 `_load_foundation`,SW143 `install_operator(graph,'T')`,SW161 source bindings의 동일 B4 grouping/uint8 RGB/registered stats로 gamma를 만들고 source graph forward를 호출한다. 그 graph를 native `core.kuramoto.prepare_coupling(...)[0]`에 전달한다. Returned A는 native symmetrization/ReLU/1e-6 floor를 포함한 actual coupling context다. Fake correlation A,learner-generated teacher A,raw unprepared graph로 대체하지 않는다. 새 context row cache는 별도 namespace/create-only이며 원본 SW161 cache를 수정하지 않는다. Q/A float32 row hashes,dtype/shape,IDs,RGB/source/operator/stats hashes와 static-setup method를 기록한다.

첫4 overlapping TRAIN images는 **원래 cache와 같은 B4** full source rollout에서 saved spikes exact replay와 실제 prepare tuple[0] capture equality를 증명한다. SW162의 B1 receipt를 B4 exact proof로 둔갑시키지 않는다.162에서 compatible B4 proof가 없으면 이 한 B4 forward만 별도로 요청한다. 모든512 A를 full-rollout-captured라고 주장하지 않고 static context reconstruction임을 명시한다. Source state는 before/after 불변이어야 한다.

## 고정 학습과 admission

SW161 `objective.mask_hungarian_loss`의 all-present-label matching CE+Dice를 모든 arm에 동일 적용한다. Named BG slot 없음. Inference GT count/matching 없음. Existing `canonical_hard_labels`의 largest hard group/min8/first-patch canonical tie 규칙을 유지한다.

512 TRAIN images,16 passes,512 updates,B16/micro4,Adam LR3e-4,global clip1. Head seed161001,message seed163001,shuffle seed161002의 동일16 permutations를 사용한다. Per logical update 네 microbatch loss를 각각1/4로 backward하고 clip/Adam은 한 번만 수행한다. Fresh optimizer initial once,transactional resume는 weights/moments/RNG/order/update를 복원한다. Core/encoder optimizer는 없다.

CPU 검사는 Q/A off-diagonal row normalization,empty row,4-channel switches,shared shapes/state,initial fresh-raw equality,Hungarian permutation invariance,largestBG/min8,real cache API를 포함한다. Actual firstB16의 두 disposable updates에서 finite P/loss/gradients,actual head update,source hash 불변을 검사한다. First output-layer credit와 second active message credit를 구분하고 inactive zero-credit는 veto하지 않는다. Nonzero trained message state에서 Q/A channel 제거가 P를 바꾸는지 기록한다. 초기 FG나 정해진 object count로 admission을 판정하지 않는다. Existing mask targets만 읽고 VAL/reserve GT는 열지 않는다.

## 고정320 평가와 결정

네 head를 위해 full source rollout을 네 번 하지 않는다. Validation image당 sourceT B1/T1024/settle512/live_tail0의 **한** untouched rollout과 prepareA capture를 공유하고,같은 actual spikes/Q/A로 네 trained head를 평가한다. Capture adapter는 original tuple을 수정하지 않고 finally 복원한다. Original sourceT QCC labels가 archived SW143T fixed320 labels와 정확히 일치해야 한다.

All4 head/sourceT/Slot predictions·protocols를 고정·검증한 뒤만 IDs1320..1639의 modal4px targets를 연다. Three metrics/largestBG policy/native32 grid를 변경하지 않는다. SourceT와 current Slot reference bytes를 재사용하고 forbidden reserve90640..90959를 읽지 않는다.

QA expansion gate: mean FG가 sourceT와 raw_only 모두보다 높고 두 shared-image paired95%CI lower bounds가 양수,foregroundIoU/objectIoU means가 sourceT 이상이어야 한다. Bootstrap10000,RNG163,동일320 sampled indices를 모든 contrasts에 공유한다. QA-vs-Q_only와 QA-vs-A_only도 prospectively 보고한다. QA>A_only CIpositive가 있어야 Q-context의 incremental benefit을 주장할 수 있지만 full gating proof는 아니다. 다른 arm이 좋아도 QA의 gate를 posthoc 통과시킨 것으로 바꾸지 않고 별도 track에 기록한다. Final512만 평가하고 best checkpoint를 GT로 고르지 않는다.

통과 후에만 source0/2 replication을 검토한다. 그 다음 별도 encoder/core joint registration,그 뒤fresh native32 70k/64를 논의한다. 이 head-only supervised result는 unsupervised .87,all3 Slot strict win,positive data scaling 또는 membrane/event/gating contribution을 충족했다고 부르지 않는다. 실패하면 K/channel/temperature/LR sweep 없이 recipe를 닫는다.
