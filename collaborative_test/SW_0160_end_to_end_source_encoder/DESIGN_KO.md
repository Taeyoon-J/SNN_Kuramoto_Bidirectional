# SW160 prospective: 실제 source RGB encoder의 end-to-end instance 관계 학습

2026-10-14. 설계 문서만 작성. SW159의 진행 코드·queue·판정은 변경하지 않는다. 새 학습은 SW159 결과가 기록된 뒤 별도 구현·검토·admission을 거친다.

## 확인한 source 경로와 가설

`snn_kuramoto_bidirectional/input_layer_generator.py::CNNFeatureEncoder`는 **activation 없는 valid 3×3 convolution**이다. 실제 trainable source 파라미터는 `kernels[8,3,3,3]`, `bias[8]`이며 native128 RGB를 `[B,8,126,126]`으로 변환한다. ReLU encoder라는 설명은 맞지 않는다.

`SW_0135_native32_spike_binding/foundation.py::encode_native32`는 uint8 RGB를 한 번 /255하고, 등록된 feature mean/std와 clip을 적용한 다음 `FeaturePatchGammaInitializer(grid_size=32)`로 adaptive-average-pool하여 `[B,8,1024]` gamma를 만든다. 이 경로에는 encoder detach가 없다. 고정 feature statistics는 재학습 중 clipping을 늘릴 수 있으므로 channel별 clipping 비율과 실제 encoder credit을 보고한다. 평가 결과로 stats를 다시 맞추지 않는다.

SW150/155/152의 deep branch는 source base encoder를 freeze했다. 이번 가설은 원래 RGB filter bank와 wider-context residual이 **같은 실제 spike 관계 target**에 공동 적응해야 한다는 것이다. 기존 residual-only 결과가 전체 source encoder의 학습 가능성을 판정하지는 않는다. 반대로 단순 filter unfreeze가 .87 FG를 달성한다는 근거도 없다.

## 하나의 matched paired pilot

seed1 source97의 strict native16→32 foundation에 SW143의 T operator를 설치한다. 초기 threshold는 source `b=0`, integration `a_d=a_m=0`이다. SW159 calibrated threshold나 SW158 amplitude adapter를 이 실험에 섞지 않는다.

양쪽 모두 기존 `SourcePreservingDeepEncoder`를 같은 RNG로 설치한다. 기존 source convolution과 zero-final-projection residual을 그대로 사용하므로 초기 feature/gamma/rollout은 sourceT와 같다.

- **control — frozen_base_joint:** source `base_encoder`만 freeze. Residual, graph의 실제 active weights, native core, 기존 integration scalar는 trainable.
- **candidate — full_encoder_joint:** 같은 파라미터와 objective에 더해 source `base_encoder.kernels/bias`를 trainable로 한다. 추가 head·decoder·external asset·mask-conditioned inference는 없다.

유일한 intervention은 source base encoder의 trainability다. 두 arm의 초기 state_dict, source order, residual RNG, optimizer recipe는 같은 기준으로 등록한다. T의 effective contrast8은 operator 내부 상수이므로 사용되지 않는 source `geodesic_contrast`는 양쪽에서 freeze하고 positive-gradient family로 잘못 요구하지 않는다.

Objective는 SW155 `objective.supervised_relation_loss`: production positive-clamped component-Pearson-product Q의 imagewise class-balanced same-instance relation MSE + positive signed-correlation loss(기존 계수 1+1). TRAIN modal32 masks만 사용한다. RGB reconstruction, source anchor, phase/global-density prior는 추가하지 않는다. 목표 정보는 GT instance 관계에서 오며 최종 assignment는 여전히 실제 emitted spikes의 QCC다.

고정 budget은 같은 source 순서의 첫2048 unique TRAIN IDs, 128 logical updates, B16/microB4, T1024/settle512/live64다. Fresh Adam은 각 arm에서 처음 한 번 생성하고 durable resume에서는 moments/RNG/update counter를 복원한다. LR: source base `3e-5`, residual `3e-6`, active graph/native core `3e-5`, 기존 integration scalar `1e-3`; global clip1. Integration은 3개 Parameter tensor / 12 scalar elements이며 중복 optimizer 등록을 금지한다. 별도 warmup·추가 image exposure는 없다.

## 최소 실행 admission

1. source strict-load 이후 T 설치, 두 encoder wrapper 초기 state 동일. 실제 고정 TRAIN B4에서 sourceT와 zero residual의 feature/gamma/six-trace/Q/QCC parity 확인.
2. 실제 두 disposable logical update. 후보의 source kernels/bias에 finite nonzero gradient와 실제 변경이 존재하고 control base는 hash 불변. Zero-output residual의 hidden weights는 첫 step에 credit 0일 수 있으므로 두 번째 step에서 검사한다.
3. Encoder→normalization→gamma→actual spike loss 경로를 detach하지 않는다. Feature statistics/patcher/normalization은 동일하고 buffer hash 불변. Clipping 비율·preclip family norms·event occupancy/transitions는 관찰값이며 임의 발화율 목표나 초기 TRAIN FG cutoff로 admission을 막지 않는다.
4. 각 arm의 checkpoint는 실제 전체 encoder(wrapper base+residual), whole wrapped core 및 Adam을 저장한다. Native validation RGB에서 **그 arm의 trained encoder gamma를 재생성**해야 한다. Source gamma cache 또는 source encoder로 후보를 평가하지 않는다.

## Endpoint와 종료

Native32 actual-spike QCC, 기존 threshold/min-group/background 규칙, IDs1320–1639의 고정320 modal4px GT 및 세 지표를 유지한다. SourceT/reference, control, candidate의 predictions를 먼저 고정·검증한 뒤 평가 GT를 읽는다. Forbidden reserve는 읽지 않는다.

seed1 pilot의 candidate FG가 sourceT 및 matched control 모두보다 개선되고 양쪽 paired-image 95% CI lower bound가 양수일 때만 동일 recipe의 seed0/2 replication을 고려한다. IoU 변화는 모두 보고하고 기존 사용자 목표/selection policy에 따라 판정한다. 비교 checkpoint를 validation으로 고르지 않고 final update128만 평가한다. 실패하면 base LR/statistics/clip/order sweep 없이 이 recipe를 닫는다. 성공적인 짧은 supervised continuation도 absolute ceiling이나 .87 보장이 아니다.

## 주장 범위와 70k/scaling 후속

이 문서는 **supervised track**이다. Source는 이미 학습된70k lineage이며 2048-image continuation이므로 fresh native32 70k 성능 또는 data scaling 증거가 아니다. Supervised recipe가 유망하면 동일 architecture의 별도 fresh native32/70k/three-seed supervised 확인을 등록한다. Unique-image 효과는 fixed-update와 fixed-epoch 비교를 분리한다. Unsupervised Slot 비교 목표를 만족했다고 바꾸어 부르지 않는다.

Encoder 학습은 saturated binary events를 자동으로 해결하지 않는다. 실제 E와 per-layer credit을 기록하고, 개선 checkpoint의 frozen dynamics ablation 및 matched retrained ablation이 있을 때만 gate 기여를 좁게 주장한다. SW159가 별도로 유용한 event regime을 찾더라도 이 paired pilot의 source를 조용히 바꾸지 않는다.
