# SW_0163_relation_context_assignment

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `c9c98d1eec7257b35f56db777a70b6d14b1a6ee4cf9d9fc5a8d2393be101d4dd`

````markdown
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

````

</details>

## OPERATIONS_NOTE.md

<details>
<summary>OPERATIONS_NOTE.md</summary>

Source SHA256: `faf57c04d3cf989f518176483d3423e5140cb6c420a4a2d056f881c347003a40`

````markdown
# Admission latency observed during the first run

The first two registered trainers eventually entered CUDA and committed real optimizer updates. Do not restart them because of startup latency or an observer timeout. See `results_archive/training_started_20261014/server_records.json` for actual process and source-bound checkpoint evidence.

Before launch, each arm deeply validates its source/context caches. The queue repeats this after GPU stabilization, then the child repeats public admission and cache constructors verify all512 rows again. During this stage parent3476869 had five threads,53338 CPU ticks and about104GB of logical reads reported by `/proc/3476869/io`. These are observations of cumulative work, not a measured standalone benchmark. The elapsed startup does not indicate a model deadlock. GPU2 correctly returned to waiting when a foreign owner appeared before launch; other users' work was untouched.

A future, separately registered operational entrypoint should perform deep CPU admission once, bind an immutable ticket to the current code/preflight/manifests/ledgers/ordered IDs and every row digest, and check small hashes plus file identities immediately under the lease. The child can then verify each row's content once at first consumption, retain that validation for later passes, and require coverage of all512 rows before final publication. This must preserve exact float32 values, four-arm initialization, training order,512 Adam updates, atomic resume and final source/input checks. Merely skipping validation or trusting a status file is insufficient.

This is a prospective optimization only. No deployed163 code, scientific recipe or existing job was changed for it. Validate a future implementation against the existing trainer's updates and serialized endpoints before using it.

````

</details>

## RECOVERY_KO.md

<details>
<summary>RECOVERY_KO.md</summary>

Source SHA256: `757a1b679589aa0168ecbc3795d5235a3edf8c00989ba196ad24550334ce8288`

````markdown
# SW0163 운영 오류 및 복구 근거

원래 학습 큐는 `failed_training`으로 종료됐고 예측·평가 큐도 선행 작업 실패로 종료됐다. 성능 평가는 아직 없다. Q_only와 A_only child는 각각512번의 업데이트를 끝내고 `training_complete`로 정상 종료했으며, raw_only와 Q_A는 학습을 시작하지 않았다.

실패는 부모 큐가 여러 스레드에서 산출물을 검증할 때 발생했다. 검증 함수가 기대 초기 모델을 다시 만드는 과정에서 `torch.random.fork_rng`와 전역 `manual_seed`를 사용한다. 이는 스레드 간에 독립적이지 않으므로, 동시에 초기화하면 기대 초기 상태 해시가 달라져 정상 manifest를 잘못 거절한다. 로컬100회 재현 모두 순차 초기화와 불일치했다. 로컬과 서버의 해시 자체를 비교한 결론은 아니다.

서버에서 기존 공개 validator를 한 스레드로 실행한 결과 Q_only/A_only **둘 다 전체 검증을 통과**했다. 두 모델의 초기 상태는 서버의 순차 초기화 값 `c9baacaa239b9ae41d913adc5ed5bee8f505df213c8b4580b24dd89ff9d15470`과 같다. 코드 fingerprint,512-update endpoint, optimizer 및 파일 해시를 보존한다. 이 두 학습을 다시 실행할 필요는 없다.

복구는 새 namespace에서 기존 실패 기록과 종료된 소유 프로세스를 확인한 뒤 두 완료 모델을 채택하고, raw_only/Q_A만 기존 trainer로 실행하는 방식이다. 부모의 모델 생성이 포함된 검증은 전부 직렬화한다. 원래 모델·loss·데이터·학습 순서·업데이트 수·실패 상태 파일은 바꾸지 않는다.

2026-10-14 서버 배포 후 실제 CPU 테스트7개가 skip 없이 통과했고, 기존 public source 준비 및 Q_only/A_only 순차 검증도 통과했다. CUDA는 초기화하지 않았으며 평가 정답은 열지 않았다. TRAIN 정답은 기존 학습 입력 검증 과정에서 재검증했다. 복구 학습/예측/평가 supervisor PID3588458/3588470/3588475를 실행하여 각 실제 `/proc/PID/cmdline`과 초기 상태를 확인했다. 등록 시 학습은 `validating_recovery`, 예측·평가는 새 선행 큐를 기다리는 상태다. 이는 큐 실행 확인이며 학습·성능 평가 완료를 뜻하지 않는다. 증거는 `results_archive/recovery_launch_20261014/server_records.json`에 보존한다.

근거:

- `results_archive/terminal_parent_validation_failure_20261014/server_records.json`: 원본 실패 상태와 완료 endpoint 해시.
- 같은 폴더의 `rng_race_diagnosis.json`: 실제 로컬 재현 범위와 플랫폼 차이.
- `results_archive/sequential_recovery_audit_20261014/receipt.json`: 서버 공개 validator 순차 검증 통과.

Goal 상태와 구분한다. 조회된 goal의 blocked 갱신 시각은1792015170, 서버 학습 큐의 실패 시각은1792016946으로 약30분 뒤다. 따라서 이 서버 오류가 앞선 goal blocked 전환 원인이라고 주장하지 않는다. Goal 도구에는 그 전환의 사유가 제공되지 않았다. 재개 후 goal 상태는 active로 확인됐다.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `675da029ecd1cb8c61d7730e6635338f5a4272604ce8ef03c0a105b646104a17`

````markdown
# SW0163 최종 결과: classifier 확장 중단

동일한 native32·fixed320·seed1 평가다. 네 모델 모두 같은 frozen source의 실제 spike와 같은 학습 순서로 512번 업데이트했다. 평가 복구에서는 추가 학습하지 않았다. 최종 예측 다섯 묶음의 public validator와 score 재계산을 통과한 뒤 두 큐가 정상 완료됐다.

| 모델 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| raw_only | 0.624734 | 0.457351 | 0.400735 |
| Q_only | 0.634939 | 0.489591 | 0.410119 |
| A_only | 0.663123 | 0.466450 | 0.434423 |
| Q_A | 0.660608 | 0.498060 | 0.432283 |
| 동일 rollout SourceT control | 0.755273 | 0.614993 | 0.579514 |
| 비교 Slot Attention seed1 | 0.841819 | 0.203417 | 0.222115 |

Q+A는 raw 대비 세 지표 모두 개선했다. FG-ARI 차이는 +0.035874, image-paired 95% CI는 [+0.017619, +0.054757]이다. A_only의 FG 개선도 +0.038389, CI [+0.019800, +0.057119]다. 이는 이 supervised classifier에서 관계 입력이 도움이 된다는 제한된 근거이며, 각 물리적 gating의 인과적 기여나 전체 모델 개선의 증명은 아니다.

Q+A는 SourceT보다 FG-ARI가 0.094665 낮고, 두 IoU도 낮다. 등록된 확장 gate는 실패했다. 이 classifier recipe의 seed0/2 복제 및 대규모 학습은 진행하지 않는다. 별도로 이미 검증된 SW165 seed1의 0.770932/0.614993/0.630523과 세 seed 평균 최고 모델을 유지한다.

다음 후보는 SW165의 foreground와 연결 partition을 유지하면서 split·merge 결정을 직접 학습하는 SW167이다. CPU/GPU Q 재계산 차이 때문에 신규 pair-feature 경로의 실제 backend 검증이 아직 필요하다. 그 검증을 통과하기 전에는 학습을 시작하지 않는다.

기존 hash-family 오류와 실패 기록을 보존했다. 복구 결과의 SHA-256은 `f189d83190e5134b00a76fb81a72202456e437d41af76233b1159c1d5575468a`다. 최종 report·두 큐 상태·실제 프로세스 확인은 `results_archive/source_lineage_prediction_frozen_20261015/public_validation_complete.json`에 있다. 이 결과는 supervised seed1 pilot이며 unsupervised 결과나 세 seed 평균으로 표기하지 않는다.

````

</details>
