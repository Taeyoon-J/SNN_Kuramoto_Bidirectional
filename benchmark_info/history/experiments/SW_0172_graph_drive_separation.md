# SW_0172_graph_drive_separation

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## ADMISSION_KO.md

<details>
<summary>ADMISSION_KO.md</summary>

Source SHA256: `df8a9a112a57a90cd6e50fc9616f03633a8ea9a158eeb7c1115d552a0f1590a4`

````markdown
# SW0172 source admission 및 matched prefix 시작

최종 v4는 실제 GPU1에서 full1024/settle512/live64를 사용해 통과했다. Original source/shared/separated의 gamma, actual normalized projection z, 실제 factorized coupling A_lat, six traces, Q, source QCC와 SW165 energy/merge-log/labels가 정확히 같았다. 원본 source 상태의 전후 SHA도 같았다.

Candidate는 context→direct drive 경로가 없고, context→실제 graph→실제 S/Q의 양의 유한 gradient를 확인했다. 이는 학습 경로의 존재를 검증하며 FG-ARI 개선이나 gate의 유용성을 입증하지 않는다. 최적화는0회, VAL RGB/GT를 읽지 않았다. CPU 단계에서 기존 TRAIN512 mask cache를 검증·소비한 범위를 숨기지 않는다.

초기 세 시도는 모델 실패 결과에서 제외한다. v1은 중첩 --worker CPU 준비 중 GPU 지정 정보가 지워져 CUDA 전에 종료됐다. v2는 source1024와 후보의 기본64 steps를 잘못 비교했고 settle512 이후가 비어 Q=0이었다. 엄격 CPU source 로드로 기본값64를 확인했다. v3는 실제 theta[B,T,N,4]를 rank3로 잘못 검사했다. 이들은 구현·검증 오류이며 모델 성능이나 routing 가설의 실패 근거가 아니다. 이전 상태/로그/소스 SHA와 원문을 모두 보존했다.

수정 후 actual native CPU1024-node/1024-step 통합 검증과 실제 --worker 인수/CUDA 지정 조건의 CPU 검증을 통과했다. 초기 검증 관련 로컬·서버21개, 학습/큐까지 합친 로컬·서버30개 검사를 통과했다.

그 뒤 original SourceT에서 fresh shared_gamma 및 separated_gamma의 matched8-update prefix를 GPU2/1에서 시작했다. 실제 PID3767717/3767705의 NVML 점유와 각 durable boundary를 확인했다. 같은 TRAIN64 두 passes/B16/micro4/Adam/고정lambda를 사용하며, shared control에 완료171 weights를 재사용하지 않는다. 완료 결과는 별도로 기록하며 이 문서의 시작 기록을 현재 진행률이나 성능 결과로 해석하지 않는다. 최고 모델은SW165다.

````

</details>

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `923cfe1b33b3a98d130c38e4ae0dbe7fb6f6994393331c1717a11df371a73336`

````markdown
# SW0172 — 학습 graph gamma와 source oscillator gamma의 분리

2026-10-15 prospective supervised DESIGN only. 코드/서버/실험을 실행하지 않는다. Sealed171은 미통과로 닫고64updates/seed/70k로 확장하지 않는다. 초기 모델은 성공한 original SourceT→SW165이며169 hard-reset/170 filter/실패한171 weights를 이어받지 않는다. Incumbent165 및 .87/세 seed/native32 fresh70k/scaling/all-gate 목표는 유지한다.

## 관찰과 단일 가설

SW171 completed prefix의 initial TRAIN8 Q/instance loss는 .3742439668/.6933890451이다. Q-only final은 .3765979657/.6973294504, Q+instance는 .3807556476/.6738497429였다. Instance supervision이 실제 z loss를 낮췄지만 actual spike Q의 task loss는 더 나빠졌다. Own-QCC nonempty는 양 arm initial/final 모두8/8이므로 sparse169의 collapse와 다르다. **8-image/8-update 결과는 장기 모델의 불가능성을 증명하지 않는다.** 단지 등록된171 continuation 조건을 미통과했다.

실제 `SW0134_native_spike_binding/rollout.py:late_rollout`의 static setup은 같은 gamma를 `graph_generator(gamma)`와 `gamma_to_drive(gamma, gamma_channel_proj, gamma_phase_gain)`에 동시에 넣는다. 한편 `graph_generator.py:125`의 actual projection z는 graph similarity와 geodesic에 쓰인다. 따라서 context 학습은 graph 표현과 oscillator의 직접 image-conditioned forcing을 **동시에** 바꾼다.

가설: z의 instance 구조를 배우는 gamma 변화가 pretrained direct-drive 표현도 움직여, graph 개선이 actual rhythm binding으로 전달되지 않을 수 있다. 이번 한 구조 변경은 **contextual gamma는 graph에만**, direct oscillator gamma는 frozen original encoder에서 공급하는 것이다. 이 경로 간섭은 아직 입증되지 않았으며 objective mismatch/작은 prefix/부족한 z 식별성 등 다른 원인이 가능하다.

## source-preserving 두 arm

두 arm 모두171의 동일 RF31 context encoder, source frozen base, normalization/statistics/clipping/pool32, actual native graph/core/12scalars를 가진다. 네 spatial conv d1/2/4/8+zero1×1→8/crop126 구조를 바꾸지 않는다. 새 head/parameter/XY/decoder/활동률 target은 없다.

같은 RGB로 `base_features=source_base(RGB)`, `context_features=base_features+context_residual(RGB)`를 계산한다. 각각 기존 mean/std/clip/pool32에 넣어 gamma_base/gamma_context를 만든다. Source base는 frozen/no running-statistics다. Base map은 한 번 계산하여 공유하는 local API를 권장한다. 두 gamma의 계산과 normalization은 양 arm에서 동일하게 실행하며 unused branch로 RNG/통계 차이를 만들지 않는다.

* `shared_gamma` matched control: graph(gamma_context), drive=gamma_to_drive(gamma_context,...).
* `separated_gamma` candidate: graph(gamma_context), drive=gamma_to_drive(gamma_base,...).

Native theta initialization도 선택된 drive를 사용한다. Graph에서 만든 actual factorized coupling을 그대로 Kuramoto가 사용한다. Core에는 phase/drive feature 외의 새 object assignment 입력이 없다. Actual emitted S→production Q→SW165 signed connected solver가 mask 경로다. Gamma_base를 readout/foreground oracle로 사용하지 않는다.

Initial context output0이므로 두 gamma 및 양 arm은 원래 SourceT와 정확히 같다. Native tau/b/emitter/reset/delays를 바꾸지 않는다. **Gamma channel projection/gain과 다른 core parameters의 trainability는171과 똑같이 유지**한다. 따라서 candidate가 학습 내내 전체 drive를 frozen하게 보존한다고 주장하지 않는다. 보호하는 것은 context residual이 direct-drive feature input을 바꾸는 경로뿐이다. 이 선택으로 비교의 유일한 차이는 gamma routing이며 hidden capacity/학습 family 차이를 만들지 않는다.

## 실제 구현 재사용 경로와 credit

`SW0171/contextual_encoder.py`, `instance_objective.py`, `train_prefix.py`의 모델/target/optimizer/checkpoint 경로를 새 leaf에서 재사용한다. Sealed files를 수정하지 않는다. New `dual_gamma_rollout`은 `SW0134/rollout.py`와 동일 loop이고 static setup의 graph_gamma/drive_gamma 인자만 분리한다. 두 tensors는 [B,8,1024],same dtype/device/source normalization이다. Source-compatible mode에서 동일 tensor를 두 인자로 전달하면 native all-six traces/Q/A와 정확히 같아야 한다.

Full1024/settle512/live-tail64를 유지한다. Graph/drive/coupling을 burn-in 전에 grad enabled로 준비하고960에서 recurrent theta/delayed theta/h/M/S만 detach한다. Candidate graph_gamma와 actual projection output은 live다. Graph 입력을 detach하거나 native setup에서 gamma_context 대신 gamma_base를 잘못 graph에 전달하지 않는다. Candidate residual→direct drive의 partial gradient는0, residual→actual native graph/actual S의 task credit는 finite actionable이어야 한다. Control에서는 source가 허용하는 두 경로가 모두 존재한다. Native graph의 actual A/gain 및 forward 실행을 캡처해 context가 실제 동역학에 쓰이는지 검증한다.

Aux z는 **actual graph(gamma_context) call**의 live projection output을 한 번 capture→native F.normalize한 값이다. 별도 graph/encoder pass나 mask head로 대신하지 않는다. LOO/BG/per-instance balancing/singleton/norm-stop 규칙은171 `instance_objective` 그대로다. Translation-equivariant z만으로 같은 appearance의 먼 instances를 구분하지 못할 수 있다는171 한계도 그대로이며 XY를 사후 추가하지 않는다.

## 동일 objective와 고정 계수

두 arm 모두171의 `L_Q + lambda*L_instance`를 사용한다. L_Q는 실제 current S의 production positive-clamped-product Q와 frozen SourceT-F/feasible targets의 constrained maximin이다. Cached Q나 z 관계를 L_Q로 대체하지 않는다. Lambda는 완료171 TRAIN-only calibration의 **0.35786938902827736**을 그대로 고정하고 original calibration/report SHA를 바인딩한다. 새 routing에 맞춰 재calibration하면 objective scale까지 바뀌므로 하지 않는다. 이 비교는 의도적으로 같은 loss 계수에서 architecture routing을 검정한다. Lambda/width/LR/clip/seed sweep은 없다.

두 arm 동일 private context seed171001/init, contextLR3e-4/graph-coreLR3e-5/integrationLR1e-3, frozen base/fresh Adam/global clip1이다. Actual parameter union은171처럼 중복 없이 구성하고 actual `dendric_layer` 이름을 포함해 family norms/deltas를 기록한다. Native E가 여전히 all-on이면 그 사실을 보고하며 memory/gate 필요성을 개선했다고 주장하지 않는다.

## 최초8 updates의 저비용 판별

Ordered512/GT TRAIN mapping은171과 동일하다. First64 두 passes=8B16updates, micro4, Random171001의171과 같은 source-index permutations를 사용한다. Actual initial/final replay는 아직 optimizer에 사용하지 않은 TRAIN positions256..263의 두 B4다. source/cache/encoder/statistics/mask shape/order hashes 및 noVAL를 유지한다.

Optimizer 전에 initial actual gamma_base/context equality, original source/control/candidate gamma/graph/A/all-six traces/Q 및 SW165 energy/log/labels exact parity를 검증한다. Same-mask/source support와 native CPU-support/GPU-energy Q 범위도 유지한다. Prefix updates1/2에서 zero-final layer→hidden layer의 credit 전개와 actual optimizer changes, routing partial gradients 및 actual Q→context via coupling credit를 확인한다. Nonfinite/dead end-to-end branch이면 중지한다. 목적 없이 disposable optimizer exposures를 추가하지 않고 actual prefix steps로 검증한다.

Prefix continuation 조건은 아래 전부다:

1. Candidate TRAIN8 actual student-Q maximin mean loss가 자기 initial보다 낮다.
2. 같은 loss가 matched shared-gamma control update8보다 낮다.
3. Candidate native-z instance mean loss가 자기 initial보다 낮다.
4. Candidate own native QCC foreground nonempty가 최소4/8이고 관련 tensors/weights/gradients가 finite다.

이는 dataset 성능의 통계적 증명이나 VAL 선택이 아니다. 고정된 작은 TRAIN probe가 noisy/짧다는 한계가 있으며, 미통과를 모델의 보편적 불가능성으로 쓰지 않는다. 미통과 시 이 recipe만 닫고 prefix/lambda/input-order를 바꿔 반복하지 않는다. Same-input/no-update control receipts를 유지한다.

Completed171 Q+instance를 shared control로 재사용할 수 있는 것은 actual init/forward/loss/order/optimizer/RNG/microbatch/truncation/runtime source가 동일함이 증명될 때뿐이다. 새 rollout의 shared-gamma mode에서 original171 parity를 검증하고 explicit adoption receipt로 이전 artifact를 읽으며 metadata를 덮어쓰지 않는다. 동등성 증명이 없으면 fresh matched shared control을 수행한다. Candidate가 새 방식이므로 실패한171 weights를 candidate에 load하지 않는다.

## 통과 후 budget와 fixed320

양 prefix gate 검증 후 candidate/control을 함께 remaining448 두 passes=56updates 더 수행한다. Total64 updates/1024 exposures/512unique이며 각 image가 두 번이다. Random171001의 original171 remaining order를 고정한다. TRAIN256..263은 continuation에 포함되며 최종 holdout이 아니다. 171 prefix control을 재사용하면 그8updates를 두 번 계산하거나 exposure를 추가하지 않는다. Logical transactional checkpoint에 두 gamma routing semantics/source/calibration/fingerprints/Adam/RNG/order/completedcounter를 바인딩한다. 실제 첫2updates latency/peak-memory를 보고한 뒤 예산을 공개한다.

Final64 두 모델은 실제 trained graph/context encoder와 해당 routing으로 gamma/drive를 생성한다. Initial/final에서 각각 (a) frozen original SourceT-F conditional diagnostic, (b) 같은 learned actual S의 native own-QCC-F standalone을 모두 평가한다. Actual learned Q/native A/gain과 SW165 induced solver는 그대로다. Initial outputs는 원래165이며 raw source/archive parity를 확인한다. F를 바꾸어 conditional을 standalone으로 표시하지 않는다.

Predictions/provenance를 모두 freeze·검증한 뒤 fixed320 IDs1320..1639/native32/modal4px 기존 세 지표를 연다. Candidate의 각 output FG가 initial/SW164 seed1 .770932099 및 해당 shared control보다 높고 각 paired95%CI lower>0이어야 한다. Conditional foregroundIoU per-image SourceT exact/objectIoU>=SW164, standalone 두 IoU means>=SW164도 필요하다. Common320 image draws/bootstrap10000/RNG172. 좋은 출력/checkpoint를 선택하지 않는다. 실패하면 recipe를 닫고 seed/70k fanout하지 않는다. 통과 때만 같은 recipe의 seed0/2를 별도 등록한다.

목표는 source의 실제 binding을 손상시키지 않는 gamma-graph 학습 가설을 검정하는 것이다. 좋은 graph embedding만으로 성공이라고 판정하지 않는다. Supervised 성과·source16-trained70k lineage·native32 continuation의 범위를 밝히고 fresh70k/positive-data-scaling/unsupervised .87/all-gate necessity는 후속 증거로 남긴다. Forbidden reserve90640..90959는 읽지 않는다. 새 generated outputs는 중앙 `collaborative_testing/prepare_output.ps1` namespace와 immutable archives에 둔다.

````

</details>

## IMPLEMENTATION_KO.md

<details>
<summary>IMPLEMENTATION_KO.md</summary>

Source SHA256: `f7471a7c24bd83945656368607d6af2bf1179d608c1f180eb18dc9fa67cd7961`

````markdown
# SW0172 구현 기준

Frozen SW0134 rollout의 동역학 loop는 그대로 호출한다. 실제 GammaToDrive nn.Module의 임시 forward pre-hook으로 첫 입력만 선택하며, 종료·예외 시 hook을 제거한다. Graph에는 context gamma를 전달한다. Base CNN은 한 번만 계산하고 두 gamma에 같은 원본 normalization/clip/native32 pooling을 적용한다.

SW165 solver가 쓰는 adjacency는 raw graph output이 아니라 실제 Kuramoto prepare_coupling의 A_lat이다. Frozen SW0162 capture helper로 이를 별도 포착하고 raw graph adjacency와 구분한다. Live normalized z는 같은 실제 graph의 단일 projection에서 포착한다.

독립 로컬 CPU 검사 6개가 통과했다. 이 중 실제 S2NetCore/PhaseStateIntegration 1024-node, 4-step/live2 구성에서 같은 gamma를 두 입력에 주었을 때 frozen rollout과 six traces/Q/z/actual coupling이 정확히 같았다. 이 짧은 검사는 pretrained SourceT의 full1024/settle512 검증을 대체하지 않는다. 아직 GPU 실험·최적화 결과는 없다.

Control은 완료171 artifact를 재사용하지 않고 fresh shared-gamma 조건을 다시 학습한다. Candidate도 원래 SourceT에서 시작하며 실패한171 가중치를 이어받지 않는다. 다음은 원래 source/shared/separated의 실제 pretrained 초기 parity와 routing credit을 검증하는 bounded GPU preflight다. 학습은 통과 후 두 조건 동일 8-update prefix로 시작한다.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `c8ac9097a64d5e426ad246bf8b2ae0599904835e12e935ea6b48897770ce8eb9`

````markdown
# SW0172 — gamma graph/drive separation prefix negative

유효한 full1024-step v4 초기 검증 후, GPU2의 shared_gamma와 GPU1의 separated_gamma가 모두 에러 없이 실제8 Adam updates를 완료했다. Original SourceT의 동일 초기 상태와 TRAIN64×2passes/B16/micro4/고정lambda를 사용한 fresh matched 비교다. Invalid v1–v3 검증 시도는 모델 실패 결과에서 제외하며 ADMISSION_KO.md에 구분했다.

| 고정 TRAIN8 진단 (낮을수록 좋음) | 초기 | shared gamma update8 | separated gamma update8 |
|---|---:|---:|---:|
| 실제 spike Q maximin loss | 0.374244 | 0.380800 | 0.380393 |
| instance prototype loss | 0.693389 | 0.673965 | 0.673843 |

경로 분리의 Q loss는 shared control보다 조금 낮지만 자기 초기보다 높다. Prototype loss 감소와 own-QCC nonempty만으로 확대 조건을 충족하지 않는다. 등록된 prefix gate가 미통과했으므로 이 recipe를64updates/추가seed/70,000장으로 확대하지 않는다. Source base/normalization/T contrast는 그대로 유지됐고 context 및 actual native parameter 변화가 확인됐다. 첫 두 실제 업데이트에서 L_Q→context gamma, Q→S 및 graph/core/context credit 검사도 통과했다.

이 결과는 짧은 TRAIN 진단의 확대 중단 근거다. VAL FG-ARI를 측정하지 않았으므로 최종 FG-ARI 하락이나 장시간 학습의 불가능성을 주장하지 않는다. Gamma 경로 분리만으로 actual spike binding을 개선하기에 충분하다는 근거를 얻지 못했다. 기존171 결과와 함께, embedding loss 감소가 task Q 개선을 보장하지 않는다는 관찰을 유지한다.

원본 final manifest/history/초기·최종 진단, checkpoint·Adam·boundary SHA, terminal PID 및 실제 배포 CPU 검증은 results_archive/completed_prefix_20261015/에 있다. 데이터·checkpoint는 GitHub에 올리지 않는다. 최고 모델은 SW165이며 전체 연구 goal은 유지한다.

````

</details>
