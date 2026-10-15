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
