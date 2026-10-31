# SW_0171_contextual_instance_graph

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## ADMISSION_KO.md

<details>
<summary>ADMISSION_KO.md</summary>

Source SHA256: `7a02efac597d0cb44763aeb44d6977b69681eb6f6ce9fcbc41328955d1b06144`

````markdown
# SW0171 학습 진입 검증 — 2026-10-15

실제 Frontier GPU1에서 source-preserving contextual encoder 진입 검증을 통과했다. 최적화 업데이트는 0회이며 성능 향상이나 최종 평가 결과가 아니다.

- Original SourceT와 zero-residual context의 동일 TRAIN RGB/RNG B4에서 gamma, native normalized z, A, 실제 Q, six traces, own QCC와 SW165 solver energy/merge log/labels가 정확히 일치했다. B1 TRAIN parity도 통과했다. VAL RGB/GT는 읽지 않았다.
- 원본 source 전체 상태와 base encoder의 전후 SHA가 일치했다.
- TRAIN positions0..15의 네 B4 평균 loss에 대해 실제 gamma gradient norm 합의 비율로 lambda=0.35786938902827736을 고정했다. Q와 instance loss 모두 양의 유한 gradient norm을 가졌다.
- 다음 학습의 고정 TRAIN8 진단은 positions256..263이며, 초기값은 첨부 independent_verification.json에 기록했다. 학습 성능 판정에 사용할 VAL 결과가 아니다.
- 로컬 및 서버 CPU 검사 26개 통과. 실제 원본 TRAIN512 캐시 소비 및 source admission도 통과했다.

첫 큐는 `python -m` 진입 시 `__name__`이 `__main__`이 되는 실행 경로를 놓쳐 child import 전에 종료됐다. 모델/CUDA/optimizer는 실행되지 않았다. 기존 실패 기록을 보존하고 canonical module 이름을 고정했으며 해당 진입 이름 회귀 검사를 추가했다. 수정 큐는 별도 v2 runtime에서 단 한 번 실행되어 정상 완료했다. preflight/model/loss는 수정하지 않았다.

원본 server 기록, terminal PID/보고서 SHA 연결 및 독립 수치 검증은 `results_archive/completed_admission_20261015/`에 있다. 다음 단계는 등록된 Q-only 대 Q+instance의 동일 8-update 비교이며, 아직 학습 결과는 없다. incumbent는 SW165 그대로다.

````

</details>

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `6c936e87ac7f42c026ef16c0cf091c0f9be6aa33e27876d2521f28af8f2f04f3`

````markdown
# SW0171 — contextual encoder + native graph instance supervision

2026-10-15 prospective supervised design. 구현/배포/실행 없음. SW169/170 sparse-hard-reset branch는 닫고 재사용하지 않는다. 시작 모델은 성공한 **원래 SourceT→SW165**다. 새로운 readout MLP/Slot/RGB decoder/외부 teacher가 아니라, actual graph를 만드는 표현을 학습한다. TRAIN GT를 사용하므로 unsupervised 성적표에 합치지 않는다. .87/세 seed/native70k/data scaling/all-gate contribution은 아직 달성했다고 주장하지 않는다.

## 근거와 가설

Source RGB encoder는 valid3×3 linear8이다. `snn_kuramoto_bidirectional/graph_generator.py:ImageConditionedGraph.forward`는 actual gamma에 `z=F.normalize(self.projection(gamma),dim=-1)`를 적용하여 cosine/geodesic graph를 만든다. 기존152/160은 source-preserving spatial residual과 consistency 또는 최종 Q pair supervision을 사용했고, 163/167은 frozen source 뒤 classifier를 학습했다. 이번 가설은 **공간 문맥이 있는 gamma와 실제 native z에 직접 instance target을 주면**, 색 재구성/self-consistency보다 객체 내부 관계를 잘 학습하고 그 graph가 원래 rhythm/spike 경로를 통해 유용한 masks를 만들 수 있다는 것이다. Shallow encoder가 .804 ceiling의 원인이라는 증명은 아니다.

## 원래 source와 정확한 capacity

Seed1 SW0097 strict source→native32 mapping→SW143-T power2/contrast8을 그대로 시작한다. `SW0135_native32_spike_binding.foundation.load_native32_foundation(1)` 및 실제 T installer를 재사용한다. Native tau/b/reset/emitter/delays/K/normalization은 원본이며169 threshold.5/slow tau/hard reset을 설치하지 않는다. Initial a_d/a_m/b도 원래 SourceT 값이다. 원본 base encoder는 **frozen**으로 보존한다.

RGB float32[0,1]에 source-preserving residual을 더한다:

```
Conv3x3(3→32,dilation1,padding1), SiLU
Conv3x3(32→32,dilation2,padding2), SiLU
Conv3x3(32→32,dilation4,padding4), SiLU
Conv3x3(32→32,dilation8,padding8), SiLU
Conv1x1(32→8,weight=0,bias=0)
residual=residual[:,:,1:-1,1:-1]
feature=original_base_encoder(RGB)+residual
```

총 spatial convolutions는 네 개, receptive field31이다. 별도 다섯 번째3×3은 없다. Crop 후8×126×126을 기존 registered mean/std/clip/adaptive_pool32에 넣는다. RGB→mask/GT-count bypass는 없다. 이 gamma가 실제 graph/Kuramoto/dendritic/membrane/actual emitted S/Q에 사용된다. `SW0152/.../deep_encoder.py`의 zero-output/crop/strict-base wrapper 패턴을 재사용하되 기존 파일은 수정하지 않고 새 opt-in leaf를 둔다.

두 arm 모두 동일 context capacity/초기 weights를 갖는다. Private initialization seed171001. 새 residual encoder, native graph(원래 T contrast는 frozen), nongraph core 및 기존 a_d/a_m/b 세 tensor=12 scalars는 trainable이다. Frozen base는 optimizer에 넣지 않는다. Fixed LR: new residual3e-4, graph/native core3e-5,integration1e-3, fresh Adam/global clip1. Parameter identity union/changed buffers를 엄격히 구분한다. Last layer0 때문에 첫 update의 hidden residual gradient0는 예상되며, 마지막 layer update 뒤 두 번째 update의 hidden credit를 확인한다.

## 두 objectives와 actual z

공통 `L_Q`는 completed168의 constrained maximin objective다. Native positive-clamped four-component product Q를 그대로 사용하고 source-F feasible targets/GT0 처리/represented-pair 조건도 그대로다. Global density/phase priors/새 threshold/decoder loss는 추가하지 않는다.

* `Q_only`: 같은 trainable context model에 L_Q만 적용한다.
* `Q_plus_instance`: L_Q+lambda*L_inst.

L_inst는 **같은 실제 graph forward에서 사용한 projection(gamma)의 normalized embedding z**에 걸린다. 두 번째 RGB/encoder/graph 호출이나 독립 mask head로 대체하지 않는다. Actual `graph.projection` forward-hook으로 그 setup의 live output을 한 번 캡처하고 native와 같은 F.normalize(dim=-1,기본eps1e-12)를 적용한다. T operator는 `_geodesic_distance`만 바꾸므로 native projection 경로를 그대로 캡처할 수 있다. Aux branch를 위해 실제 graph input/projection value를 detach하지 않으며 finally hook을 해제한다. 실제 native normalization과 같은 값/차원/eps를 사용하며 원본 forward trace를 바꾸지 않는다.

## Leave-one-out spherical prototype CE

각 TRAIN image의 **전체1024 modal32 labels**를 사용한다. Frozen source-F 안만 쓰지 않는다. GT ID0도 한 background class이며 별도 weight를 주지 않는다. Image 안의 visible GT IDs는 TRAIN target/prototype 구성에만 쓰이고 inference에는 전달하지 않는다.

Class k의 members Ck, normalized z_i에 대해 other-class prototype mu_l=normalize(mean_{j in Cl} z_j). Query i의 own prototype만 mu_(k,-i)=normalize((sum_{j in Ck}z_j-z_i)/(|Ck|-1))로 정의한다. Query CE는 actual z_i와 these prototypes의 dot product/.1 logits로 own class를 맞힌다. Temperature .1은 사전 고정이며 sweep하지 않는다. Prototype sums/means는 live다.

|Ck|=1인 class의 **queries는 제외**하지만 유효한 singleton z는 다른 query의 negative prototype으로 남는다. 최소 두 visible classes가 없거나 valid query class가 없으면 image L_inst=0이다. 그 외에는 각 valid-query class의 mean CE를 구한 뒤 class means를 동일 weight로 평균한다. BG patch가 많아서 loss를 지배하지 않는다. B4 loss는 zero-loss image도 포함한 image mean이다. Excluded queries/classes/image counts를 기록한다.

Valid query embedding의 pre-normalization norm, 필요한 full/LOO prototype norm이 native eps(1e-12) 이하이거나 nonfinite면 중지한다. 새로운 eps/noise/relabeling으로 구제하지 않는다. Translation-equivariant RF31 gamma/z만으로 동일하게 보이는 먼 두 instance를 구분하지 못할 수 있다. 실제 graph A에는 spatial/geodesic position 정보가 있지만 z auxiliary 자체에는 없다. Zero CE 또는 모든 same/different pair의 완전 분리를 가정하지 않는다. 이 위험을 이유로 결과를 본 뒤 XY/새 head를 추가하지 않는다. Prototype CE도 collapse와 object identifiability를 보장하지 않으며 actual-Q task 개선을 별도로 요구한다.

## Lambda — optimizer 이전의 TRAIN-only calibration

같은 ordered512의 positions0..15, **첫 네 B4**를 양 arm 공통 초기 모델에서 no-optimizer gradient calibration에 쓴다. 각 B4 image-mean L_Q/L_inst의 실제 gamma에 대한 gradients gQ_b/gI_b를 계산한다. Image loss0인 경우도 평균과 gradient에 그대로 포함한다; 미리 제외해 비율을 높이지 않는다.

`lambda=(sum_{b=0..3} ||gQ_b||_2)/(sum_{b=0..3} ||gI_b||_2)`.

Norm은 동일 live gamma tensor 전체의 L2이며 finite-positive 두 sums를 요구한다. Dead sum/nonfinite면 stop; lambda clipping/다른 batches/ratio/epsilon으로 rescue하지 않는다. Gradient-only calibration은 optimizer exposure가 아니지만 **16 TRAIN images와 TRAIN GT를 사용했다는 비용**을 명시한다. 동일 lambda를 저장하고 재계산/VAL 선택하지 않는다. Control은 같은 gamma/L_inst를 진단용으로 계산할 수 있으나 gradient objective에는 lambda 항을 넣지 않는다.

## Source parity·시간 경로·최소 credit 확인

Original SourceT와 zero-final context model을 실제 같은 TRAIN RGB/RNG/B4에서 비교한다. Gamma/base maps/graph와 actual A, all-six source traces/Q 및 원래 SW165 energy/merge-log/labels가 같아야 한다. B1도 같은 TRAIN RGB에서 SourceT/SW165의 실제 forward/readout 경로를 비교하며, archived SourceT/SW164 protocol·source fingerprint는 별도로 대조한다. TRAIN trace가 다른 VAL image의 archived bytes와 같다고 주장하지 않는다. 이 진입 검증에서 VAL RGB/GT는 읽지 않는다. 단순 frozen encoder state 비교로 forward parity를 대신하지 않는다. 기존 CPU-support vs GPU-energy Q/cache-layout 모드 구분은 유지한다.

`SW0134_native_spike_binding.rollout.late_rollout`의 full1024/settle512/live-tail64를 사용한다. Actual gamma drive/graph/coupling과 aux z는 grad enabled 상태에서 burn-in 전에 준비한다. Boundary960의 recurrent state만 기존처럼 detach한다. Aux z가 no_grad burn-in에 휩쓸리거나 record용 detach가 되지 않아야 한다. Whole512 actual S로 native Q를 계산한다. Existing `SW0168/train.py`의 RGB data path, native family/checkpoint/batch evaluator 골격을 재사용하고 sealed source는 변경하지 않는다.

Prefix의 첫2 실제 updates에서 finite loss/actual Q→S/encoder/graph credit, output→hidden residual credit 전개와 actual changed params를 확인한다. 이름은 `dendric_layer`를 포함한 실제 parameter set을 사용하여 Kuramoto/dendrite/membrane/a_d/a_m/b를 분리 기록한다. M/E credit가 작거나 E가 계속1이면 그대로 보고하며 인위적 update/활동률 목표로 통과시키지 않는다. 모든 핵심 end-to-end family가 dead/nonfinite이면 stop한다. Source initial parity와 candidate/control init identity는 필요하지만 학습 후 source 동등성은 요구하지 않는다.

## 정확한 exposure 순서·stage gate

두 arm은 source seed1의 같은 registered ordered512 TRAIN RGB/mask IDs를 쓴다. Prefix는 **처음64개를 두 complete passes**: 각 pass의 source positions0..63 permutation을 Random171001의 연속 draws로 만들고 B16으로 나눈다. 총8 updates/128 optimizer exposures다. 같은 image는 두 번씩이고 candidate/control 순서는 같다.

Prefix 직전 initial 및 update8 뒤 **positions256..263**의 fixed TRAIN8을 no-update로 replay한다. 이8은 prefix optimizer에 사용하지 않았지만 Stage2에는 포함된다. Calibration도0..15뿐이므로 이8은 calibration에도 미사용이다. VAL holdout이나 최종 generalization proof로 부르지 않는다.

진행 조건은 candidate의 이8 actual student-Q maximin image-mean loss가 (a) 자기 initial, (b) 동일 control update8보다 각각 낮고, L_inst image-mean도 자기 initial보다 낮으며, own native QCC foreground nonempty가 최소4/8인 것이다. L_Q는 cached Q나 z/prototype 관계로 대체하지 않는다. Frozen original SourceT-F에서 계산한 feasible-target loss이며 own student Q를 입력한다. All declared inequalities are strict; candidate/control/initial은 같은 B4 grouping이다. Finite actual families와 정상 checkpoint도 필요하다. 하나라도 미달이면 prefix 결과를 보존하고 이 recipe를 닫는다. LR/lambda/kernel/sample/loss cutoff를 조정해 반복하지 않는다.

통과하면 positions64..511의 **나머지448개를 두 complete passes**한다. 같은 RNG171001의 다음 draws로 각 pass permutation을 만들고56 updates를 더한다. Prefix checkpoints/Adam/RNG에서 그대로 이어간다. 총64 updates/1024 exposures/512 unique, 모든 image가 정확히 두 번이다. **기존169 shuffle과 같다고 주장하지 않는다.** TRAIN8은 continuation에 포함되므로 최종 held-out test가 아니다. Candidate/control 중 좋은 쪽만 continuation하지 않는다; 두 arm의 gate/완주·실패를 모두 보고한다. Logical B16 micro4/image-mean/clip-Adam once다. 실제 첫2 prefix updates의 latency/peak memory로 remaining budget을 계산하고 공개한다.

추가 disposable optimizer updates를 숨기지 않는다. Code preflight는 init을 검증하고, 학습 credit 검증에는 실제 prefix updates1/2를 사용한다. Calibration gradient calls, initial/prefix replays 및 final evaluations의 추가 forward 비용은 optimizer 노출과 분리 보고한다. 각 update에 source/init/lambda/calibration/order/RNG/Adam/completedcounter/runtime을 묶어 transactional resume한다.

## 최종 fixed320의 두 출력·learning effect

Optimizer0 candidate/control은 같은 forward이며 한 actual rollout을 공유해 initial predictions를 만든다. Final64 candidate/control은 각 trained **encoder를 실제 사용하여 gamma를 재생성**한다. 각 endpoint는 (1) frozen original SourceT-F conditional diagnostic, (2) 같은 learned rollout의 own CPU QCC-F standalone을 모두 생성한다. 두 energy는 learned actual GPU Q/native A/gain의 SW165 solver다. Conditional F를 learned own F로 몰래 대체하지 않는다. Inference에 prototype/GT/scene count를 쓰지 않는다.

Initial/final·두arm·두outputs와 SW164/SourceT/Slot references의 predictions/own manifests를 전부 freeze·검증한 뒤 fixed320 IDs1320..1639/native32/modal4px 기존 세 지표를 계산한다. VAL를 prefix go/모델/계수 선택에 쓰지 않는다. Reserve90640..90959는 읽지 않는다.

Final candidate의 각 output FG가 initial, SW164 seed1 .770932099 및 해당 final control보다 높고 각각 paired95%CI lower>0이어야 한다. Conditional foregroundIoU per-image SourceT exact/objectIoU>=SW164, standalone 두 IoU means>=SW164도 요구한다. Bootstrap10000/shared320 draws/RNG171. 좋은 head/output/checkpoint만 골라 승리로 부르지 않는다. 실패하면 recipe를 닫고 sweep/seed/70k fanout하지 않는다. 통과 때만 동일 recipe의 seed0/2를 등록한다.

원래 native event가 계속 all-on일 가능성은 남아 있다. Aux가 graph 표현을 개선해도 모든 gate의 유용성은 증명되지 않는다. Trained model의 실제 dynamics ablation과 필요시 matched retraining은 별도다. Three-seed .87/Slot all3 strict win/fresh native32 70k/positive unique-data scaling 및 unsupervised 주장도 별도 증거가 필요하다. Source code는 canonical source folders에, 새 generated outputs는 `collaborative_testing/prepare_output.ps1` 중앙 namespace와 immutable archives에 둔다.

````

</details>

## INPUT_CACHE.md

<details>
<summary>INPUT_CACHE.md</summary>

Source SHA256: `2f0a472cde1e08da2833d0d87b2b16cbb214632c7523b7ccc99498406fd87a98`

````markdown
# SW0171 verified reusable TRAIN inputs

2026-10-15. The CPU-only producer3736173 completed normally in374.99 seconds.
It materialized the same original-source TRAIN512 RGB, source foreground support,
feasible labels and canonical full modal32 labels once for both comparison arms.
It did not initialize a candidate, run CUDA or update parameters. No images or
labels were generated or changed, and validation/reserve rows are absent.

Root independently re-read all512 canonical HDF5 RGB/mask rows, recomputed their
modal32 and feasible targets, and compared every cached row. All data hashes,
schema, original source order, producer code/dependencies and support bindings
matched. The actual arrays remain exclusively on Frontier. Git contains only
the producer text and validation metadata under input_cache_ready_20261015.

Consumers can validate immutable bytes/source/HDF identity and reuse these exact
inputs; they should not recompute the full CPU QCC cache separately for every arm.
The producer's SW169 metadata describes data provenance only. SW171 uses original
SourceT model dynamics, not SW169 learned parameters, slow membrane or hard reset.

Manifest SHA256:
`388e8748f5ebf7e8155e95b0651ddc59f239a467ea9351f26d2f6b16cebb8a67`.
Arrays SHA256:
`4006f191baff3b9e1c9effc8ec67421c011726a5b02b05fdb8c9a432499819ba`.
This is verified input preparation, not a SW171 segmentation score or GPU test.

````

</details>

## LEAF_IMPLEMENTATION.md

<details>
<summary>LEAF_IMPLEMENTATION.md</summary>

Source SHA256: `f79f61d174dffa1220607f502dcca2e45c8bbfee18afe796094344883d0b917d`

````markdown
# SW0171 pure encoder and objective implementation

2026-10-15. Contextual encoder and leave-one-out instance objective are implemented.
Root and the independent reviewer each passed all 12 CPU contracts using the
actual native CNNFeatureEncoder, encode_native32 and graph projection APIs.
The zero-output residual preserves initial feature maps and gamma exactly.
The original base kernels stay frozen; the contextual branch remains trainable
jointly with the graph/core in the registered forthcoming training comparison.
Constructor seeding preserves caller RNG. Hidden residual credit is expected to
appear after the initially zero final layer changes; tests check this behavior.

The auxiliary loss uses the actual graph projection output captured exactly once,
with native normalization, live leave-one-out prototypes and class-balanced CE.
Tests cover gradients, background/instance-label exchangeability, singleton
queries versus negative prototypes, zero-loss images and uniform exclusion counts.
These CPU checks prove implementation behavior, not object segmentation gains.

No SW171 GPU model experiment or optimizer update has run at this stage. The
separate CPU-only original-source TRAIN512 cache producer was observed live as
PID3736173; a launch or lock alone is not proof it completed. It reuses verified
source inputs and will save canonical full modal labels separately from feasible
source-support targets. It does not initialize the failed SW169 model or change
the dataset. Dataset/cache arrays remain on Frontier and are not published.

Preflight source parity, lambda calibration, actual eight-update comparison,
fixed320 performance and all broader gates still require runtime evidence.
SW165 remains the incumbent. Source files and generated outputs retain their
existing separate canonical/centralized locations.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `38ec5f62c00bfab703f2dec89e779a4b2e5a344a24d5382e9625596379b84cf0`

````markdown
# SW0171 — matched prefix negative

두 조건 모두 GPU1/2에서 에러 없이 실제 8 Adam updates를 완료했다. 동일 source 초기화, 동일 TRAIN64 두 pass와 B16/micro4, 각 128 exposures를 검증했다. 첫 두 업데이트의 Q→S 및 graph/core/context credit과 실제 가중치 변화가 확인됐고, frozen base/preprocessing/T contrast는 유지됐다. 학습 후 original source 전체 digest가 바뀌는 것은 trainable core가 실제 변경됐기 때문이다.

| 고정 TRAIN8 진단 (낮을수록 좋음) | 초기 | Q-only update8 | Q+instance update8 |
|---|---:|---:|---:|
| actual-spike Q maximin loss | 0.374244 | 0.376598 | 0.380756 |
| instance prototype loss | 0.693389 | 0.697329 | 0.673850 |

Q+instance는 embedding의 물체별 분류 loss를 낮췄지만, 실제 spike에서 계산한 연결 loss는 초기 및 Q-only보다 모두 나빠졌다. 따라서 등록한 prefix gate를 통과하지 못했으며, 이 recipe를 64 updates/추가 seed/70,000장으로 확대하지 않는다. own-QCC nonempty 통과 여부 및 모든 소수점 원본은 첨부 독립 검증에 기록했다.

이 결과는 짧은 TRAIN 진단에서의 확대 중단 근거다. VAL FG-ARI를 측정하지 않았으므로 최종 FG-ARI 하락이나 장시간 학습의 불가능성을 주장하지 않는다. 또한 encoder gradient/가중치 변화만으로 gate의 물체 구분 기여가 입증된 것은 아니다. 현재까지의 증거는 embedding 개선이 actual-spike binding 개선을 보장하지 않는다는 것이다.

`results_archive/completed_prefix_20261015/`에 두 최종 manifest/history/초기·최종 진단, checkpoint·Adam·boundary SHA, terminal PID와 실제 배포 테스트 기록 및 독립 재계산을 보관했다. 데이터·checkpoint 파일은 GitHub에 올리지 않는다. 최고 모델은 여전히 SW165이며 goal은 유지한다.

````

</details>

## TRAINING_KO.md

<details>
<summary>TRAINING_KO.md</summary>

Source SHA256: `00ad3869517bf15f5161626a3ca0c36351b02b59204032d2441f223fd552062d`

````markdown
# SW0171 matched prefix — 2026-10-15

동일한 original SourceT 초기 상태와 zero-final RF31 context로 Q-only와 Q+instance 두 조건의 학습을 시작했다. 실제 worker PID 3741306/GPU1 및 3741314/GPU2의 NVML 점유와 각 durable boundary 생성을 확인했다. 다른 사용자의 GPU0 작업은 건드리지 않았다.

두 조건 모두 첫 TRAIN64 RGB/정답 mask를 동일 순서로 두 번 학습한다. 각 조건은 logical B16/micro4의 8 Adam updates, 128 exposures다. Frozen base CNN과 preprocessing/T contrast는 유지하고 contextual encoder, graph와 native core, integration scalars를 학습한다. Q+instance만 같은 실제 graph projection의 leave-one-out class-balanced prototype CE를 추가하며, 사전 TRAIN-only calibration lambda=0.35786938902827736을 고정한다. 이 실험은 supervised TRAIN-mask 조건이다.

- 실제 full1024/settle512/live-tail64 rollout의 component spikes로 기존 production Q를 만든다. RGB-to-mask 우회 head는 없다.
- 첫 두 실제 업데이트에서 Q→S, graph/core 및 contextual output/hidden layers의 학습 credit을 검사한다. 실패하면 그대로 terminal scientific rejection으로 기록하며 추가 노출 없이 멈춘다.
- 학습 전후 고정 TRAIN positions256..263에서 actual-Q loss, instance loss와 own-QCC nonempty를 비교한다. 아직 VAL FG-ARI나 70,000장 개선을 입증한 결과는 아니다.
- Prefix가 등록된 비교 기준을 통과한 경우에만 나머지448 images 두 pass를 이어간다. incumbent는 아직 SW165다.
- 로컬·서버 CPU 검사 38개 및 실제 terminal admission→TRAIN cache 소비 경로가 통과했다. Adam/boundary 복원, initially-frozen source의 명시적 trainable-family 활성화, 중단 판정의 crash/resume 안전성을 확인했다.

현재 실행 증거는 `results_archive/started_prefix_20261015/actual_start.json`이며 완료 결과와 구분한다. Frozen 코드와 실행 근거를 공유하며, 데이터와 checkpoint 파일은 GitHub에 올리지 않는다.

````

</details>
