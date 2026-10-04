# patch_v2_sw 새 대화 인수인계 문서

작성 기준일: 2026-10-04 (America/New_York)

이 문서는 매우 길어진 기존 대화를 닫거나 삭제하지 않은 채, 새 대화가 현재 연구를 거의 정보 손실 없이 이어받도록 만든 권위 있는 인수인계 자료다. 새 대화의 에이전트는 이 문서를 출발점으로 사용하되, 실행 전에는 반드시 실제 Git 상태, `collaborative_test/STATUS.md`, 서버 프로세스와 최신 동료 브랜치를 다시 확인해야 한다.

## 1. 현재 작업의 정체성과 범위

- 현재 핵심 프로젝트는 `patch_v2_sw`다.
- 로컬 저장소: `C:\Users\SangWoo Kim\OneDrive - University of North Carolina at Chapel Hill\Desktop\ACMLab\patch_v2_sw`
- GitHub 작업 브랜치: `patch_v2_sw`
- 동료 브랜치: `patch_v2`
- 서버 작업 복사본: `/Data0/kevinswk/patch_v2_sw`
- 2026-10-04 확인 시 로컬 최신 커밋: `28d24b0 Diagnose spatial prior versus learned membrane signal`
- 당시 `git status --short`는 비어 있었으므로 작업 트리는 깨끗했다.
- 동료 브랜치에서 마지막으로 확인된 커밋: `b99fac0 Record the resume point, and queue experiments server-side`
- 현재 진행 중이라고 확인된 서버 학습/평가 프로세스는 없다.
- 이전 goal 상태는 일시 정지 상태다. 새 대화에서는 동일한 목표를 새 goal로 명시적으로 다시 시작해야 한다.

### 혼동하면 안 되는 이전 프로젝트

- `patch_sw`: 단일 16×16 patch grid, dense/dendritic/membrane/spike loss와 classifier를 여러 차례 실험했던 이전 계열이다.
- `u_net_sw`: U-Net의 4개 feature level을 각각 별도 S2Net core로 처리하고 coarse-to-fine spike 경로를 따라 mask를 만드는 다중 레벨 계열이다.
- `patch_v2_sw`: 현재 goal-mode의 권위 있는 프로젝트다. 사전 계산된 gamma sequence와 learned image-conditioned graph, Kuramoto, dendritic SNN, spike/membrane readout을 사용한다.
- 이전 프로젝트의 아이디어와 진단 결과는 참고할 수 있지만, 파일·checkpoint·평가 split·점수를 현재 `patch_v2_sw` 결과와 섞어서는 안 된다.

## 2. 사용자가 원하는 최종 목표

최종 예측은 반드시 모델이 생성한 **spike 또는 membrane history에서 만든 object mask**여야 한다. theta 또는 PLV clustering은 학습 loss나 진단에는 사용할 수 있지만 최종 mask로 직접 사용하면 안 된다.

고정된 동일 평가 계약에서 seed 0, 1, 2로 학습한 우리 모델의 세 지표 평균이 비교 가능한 Slot Attention의 세 지표 평균을 모두 엄격하게 초과해야 한다.

1. `patch_fg_ari`
2. `patch_foreground_iou`
3. `patch_matched_object_iou`

단일 실험이 세 지표를 모두 올릴 필요는 없다. 다음 세 트랙을 유지한다.

- FG-ARI 우선 트랙: foreground object끼리의 분리가 좋아지는 방법을 보존한다.
- foreground/mask quality 트랙: foreground IoU 또는 matched-object IoU를 올리는 방법을 보존한다.
- balanced 트랙: 동일 baseline에 대해 세 지표를 모두 올리는 방법을 최우선 후보로 둔다.

최종 성공 선언은 validation에서 고른 설정을 고정한 뒤, 동일한 test split과 동일한 평가 함수에서 3-seed 평균으로 확인할 때만 가능하다.

## 3. 사용자가 명시한 연구 및 작업 원칙

1. Core의 기본 계산 순서와 모델 구조는 최대한 유지한다.
2. hyperparameter, loss, classifier는 자유롭게 개선할 수 있다.
3. 가능하면 실제 spike 결과에 영향을 주는 loss를 사용한다.
4. Kuramoto theta가 의미 있는 위상 동기화 구조를 만드는지 PLV와 GT 진단으로 확인한다.
5. theta → sinusoidal gating → dendritic state → membrane → spike로 정보와 gradient가 실제로 전달되는지 확인한다.
6. classifier를 무작정 sweep하지 않는다. 먼저 물체 수와 spike/membrane pattern grouping을 합리적으로 처리할 기준 classifier를 만든다.
7. 각 단계의 activation을 GT patch mask와 비교하여 object separability가 어느 단계에서 사라지는지 찾는다.
8. 병목을 진단한 뒤에만 loss·hyperparameter·classifier를 바꾸고 고정 split에서 다시 평가한다.
9. 한 지표가 떨어졌다는 이유만으로 유용한 아이디어를 즉시 버리지 않는다. 개선된 지표와 후속 결합 가능성을 기록한다.
10. major 변경, 의사 결정이 필요한 내용, 실제 개선이 확인된 경우만 사용자에게 보고한다. routine epoch 진행 보고는 하지 않는다.
11. 각 test가 무엇을 바꿨는지 쉬운 말로 자세히 설명하고, 결과도 읽기 쉽게 기록한다.
12. 코드가 바뀌면 파일, 함수, 이전 동작, 새 동작, 변경 이유를 정확히 기록한다.
13. 실험 결과와 insight는 `collaborative_test`에 번호를 붙여 저장하고 `patch_v2_sw`에 commit/push한다.
14. 동료의 `patch_v2` 브랜치도 매 cycle 확인하되, 평가 계약과 데이터가 다른 결과를 숫자로 직접 합치거나 코드를 맹목적으로 merge하지 않는다.

## 4. 모델의 이론적 목적

이미지의 각 patch를 하나의 oscillator로 보고, 이미지에서 만들어진 gamma가 oscillator의 외부 drive와 초기 phase를 만든다. image-conditioned graph는 patch 간 coupling을 만든다. Kuramoto dynamics가 object에 속하는 patch들이 유사한 phase 관계를 갖도록 유도하고, sinusoidal gating이 위상 궤적을 SNN 입력으로 변환한다. Dendritic layer가 phase-derived 입력을 시간적으로 누적하고 membrane layer가 이를 membrane potential과 spike로 바꾼다. 마지막 classifier가 patch별 spike/membrane history를 object instance label map으로 묶는다.

핵심 연구 가설은 다음과 같다.

- 같은 object patch는 색·위치·feature와 graph coupling 때문에 위상과 activation pattern이 비슷해질 수 있다.
- 다른 object patch는 낮은 coupling 또는 경계 정보 때문에 다른 위상/activation pattern을 가져야 한다.
- 이 분리가 membrane과 spike까지 보존된다면, 최종 classifier는 GT object 수를 직접 알지 않고도 patch들을 object별로 묶을 수 있다.

현재 최신 진단은 이 가설이 아직 충분히 실현되지 않았음을 보여준다. 현재 최고 FG-ARI classifier의 이득 대부분은 learned membrane signal보다 spatial Gaussian prior가 설명한다. 따라서 다음 단계는 classifier 미세 조정만 계속하는 것이 아니라 membrane/spike에 실제 object identity가 더 강하게 나타나도록 학습 신호와 전달 경로를 진단·개선하는 것이다.

## 5. 현재 코드 기준 모델 계산 경로

주요 구현은 `snn_kuramoto_bidirectional/s2net_cls.py`에 있다.

### 5.1 이미지 또는 저장 gamma

- end-to-end wrapper인 `S2NetClassifier`는 `GammaGenerator`와 `S2NetCore`를 가진다.
- 현재 goal-mode 학습은 주로 이미지에서 gamma를 즉시 만들지 않고, 저장된 gamma sequence 파일을 `train_s2net_core.py --gamma-seq-path`로 읽어 core를 학습한다.
- gamma shape의 대표값은 `[B, C, N] = [B, 8, 256]`이다.
- `C=8`은 feature-map/channel 수, `N=256`은 16×16 patch oscillator 수다.

### 5.2 GammaToDrive

- 구현: `gamma_initializer.py`의 `GammaToDrive`.
- readout 기준은 `gamma_drive_mode=static`이다.
- `gamma_channel_proj = Linear(8, osc_dim=4, bias=False)`가 각 patch의 8-channel gamma를 4차원 Kuramoto drive로 투영한다.
- weight는 생성 시 `normal_(std=1/sqrt(8))`로 초기화되고 학습 parameter다. 이는 forward마다 실행되는 정규화가 아니라 constructor에서 단 한 번 실행되는 초기화다.
- `gamma_phase_mode=standardize_tanh`이면 sample 내부 gamma/drive를 표준화한 뒤 `pi * tanh(gain * gamma)`로 phase 범위에 매핑한다.
- `theta_init=gamma`이면 이 drive가 초기 theta가 된다.
- static 모드에서는 같은 drive가 64개의 Kuramoto step마다 반복 입력된다.

### 5.3 Image-conditioned graph와 SC

- 구현: `graph_generator.py`의 `ImageConditionedGraph`.
- `graph_mode=learned`이면 gamma sequence에서 sample별 `[B,N,N]` coupling graph를 forward당 한 번 생성한다.
- 기본 시작 설정은 `graph_top_k=32`, `graph_spatial_decay=0.55`다.
- feedback strength가 0이면 매 timestep graph를 다시 만들지 않는다. 동일 forward에서 생성된 graph를 rollout 전체에 사용한다.
- static SC는 `self.sc` buffer를 batch로 expand하지만, 현재 기본 시작 설정은 learned graph다.

### 5.4 Kuramoto dynamics

- 구현: `kuramoto_layer.py`의 `graphVectorKuramoto`.
- 각 step에서 이전 theta, 현재 static drive, graph adjacency를 받아 새 theta를 만든다.
- 현재 기준 `N=256`, oscillator 내부 vector dimension `D=4`, coupling `k=256`, `freq_gain=2.0`이다.
- rollout은 64 timestep이다.
- PLV는 theta 또는 membrane/spike-derived signal의 pairwise synchrony를 진단하거나 loss로 사용한다.

### 5.5 Sinusoidal gating

- 구현: `sinusoidal_gating.py`의 `sinusoidal_gating`.
- delayed phase mean으로 `mask = 0.5 * (1 + sin(delayed.mean(D)))`를 만든다.
- `gate_mode=raw`에서는 mask를 다시 sigmoid로 압축하지 않는다.
- 일반 모드에서는 `sin(theta) * mask`; `phase_mean` 모드에서는 component를 먼저 평균해 하나의 sinusoid를 만든다.
- 현재 시작 설정은 `gate_mode=raw`이고, `spike_per_component`를 사용할 때 4개 component를 fold하여 shared dendritic/membrane layer에 통과시킨다.

### 5.6 Dendritic layer

- 구현: `dendric_layer.py`의 `DendricLayer`.
- phase-derived wave를 여러 dendritic branch가 temporal state로 누적하고 합쳐 `h_wave`를 만든다.
- 과거 patch_sw 진단에서는 branch 상쇄, `sum(abs(branches))` 대비 `abs(sum(branches))`, dense magnitude가 중요한 병목 후보였다.
- 현재 patch_v2 goal에서는 core architecture를 크게 바꾸기 전에 실제 gradient와 object separation을 단계별로 진단해야 한다.

### 5.7 Membrane과 spike

- 구현: `membrane_layer.py`의 `MembraneLayer`.
- `h_wave`와 gate를 사용해 membrane state를 업데이트하고 threshold를 넘으면 spike를 만든다.
- 시작 설정: `membrane_vth=0.06`, `membrane_low_m=-4`, `membrane_high_m=0`.
- `spike_per_component`에서는 component별 membrane/spike를 계산한 후 component 평균을 일반 `core_out`과 `spikes`로 사용하며, component-level tensor도 `last_component_out`, `last_component_spikes`에 보존한다.

### 5.8 Core output과 classifier

- `core_out`: 대체로 `[B,N,T]` membrane history.
- `spikes`: `[B,N,T]` aggregate component spike history.
- `theta_stack`: 요청 시 `[B,T,N,D]`.
- `S2NetCore._detect_object_groups`는 `spike_rhythm`, `spike_interval`, `spatial_components` 중 설정된 classifier를 호출한다.
- 실제 goal 평가에서는 이 내부 기본 classifier 외에도 `collaborative_test`의 spectral/adaptive-slot/component classifier들을 frozen checkpoint output에 적용했다.
- 최종 evaluator input은 16×16 patch label map이어야 한다. overlapping masks와 unassigned patch를 GT 없이 결정해야 한다.

## 6. 학습 loss의 현재 의미

기본 시작 설정은 phase PLV 계열 loss를 사용하고 spike-rate/smooth/diversity/structural weight를 0으로 둔다.

- `plv_collapse_loss`: 모든 pair가 비슷한 PLV로 collapse하는 것을 방지한다.
- `plv_bimodality_loss`: pairwise PLV가 애매한 중간값보다 낮거나 높은 두 모드로 갈리도록 유도한다.
- `plv_group_balance_loss`: 높은 PLV edge density를 target density 근처로 유도한다.
- `plv_spatial_coherence_loss`: 공간적으로 인접한 patch의 구조적 coherence를 유도한다.
- `plv_source=phase`이면 loss가 theta에 직접 걸리므로 dendritic/membrane parameter에는 gradient가 연결되지 않을 수 있다.
- SW_0002/0003에서 `plv_source=membrane`을 시험한 이유가 theta의 구조를 membrane까지 전달하도록 SNN 경로를 직접 학습하기 위해서였다.
- RGB edge membrane separation loss도 존재하지만, SW_0016의 weight 0.1은 grouping 성능을 악화시켰다. 더 작은 weight를 무조건 채택해서는 안 되며, 기존 결과와 matched ablation으로 판단해야 한다.

## 7. 기본 시작 학습 설정

다음은 사용자가 지정한 최고 시작 설정이다. 서버 경로는 우리 환경에 맞춘다.

```bash
python -u -m snn_kuramoto_bidirectional.training.train_s2net_core \
  --gamma-seq-path /work/USERS/tkim1/gamma_sequences/wm_patch_gamma_seq_k8_grid16.pt \
  --num-regions 256 --num-feature-maps 8 --device cuda \
  --epochs 40 --batch-size 16 --lr 1e-3 --seed 0 --osc-dim 4 \
  --gamma-drive-mode static --num-time-steps 64 --plv-settle 32 \
  --theta-init gamma --gamma-phase-mode standardize_tanh --freq-gain 2.0 \
  --graph-mode learned --graph-top-k 32 --graph-spatial-decay 0.55 \
  --spike-spatial-grid-size 16 --k 256 \
  --membrane-vth 0.06 --membrane-low-m -4 --membrane-high-m 0 \
  --low-n -4 --high-n 0 --gate-mode raw \
  --plv-source phase --plv-combine mean --loss-signal sigmoid_membrane \
  --spike-rate-weight 0 --spike-smooth-weight 0 \
  --spike-diversity-weight 0 --structural-weight 0 \
  --plv-collapse-weight 1.0 --plv-bimodality-weight 1.0 \
  --plv-balance-weight 10.0 --plv-target-density 0.867 \
  --plv-coherence-weight 0.5 --spike-per-component --verbose \
  --save-path "$RUN_DIR/core.pt"
```

`--spike-per-component`는 spike readout을 보기 위한 시작 설정에 포함한다. 경로와 GPU 번호는 실행 직전에 확인한다.

## 8. 데이터와 평가 계약

- CLEVR HDF5: `/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5`
- train gamma: `/work/USERS/tkim1/gamma_sequences/wm_patch_gamma_seq_k8_grid16.pt`
- full evaluation gamma: `/work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt`
- train IDs: 0–999
- validation IDs: 1320–1639, 320 images
- reference test IDs: 1000–1319, 320 images
- image size: 128×128
- patch size: 8×8 pixels
- patch grid: 16×16
- oscillators: 256
- GT HDF5 mask는 128×128 integer instance IDs다.
- `clevr_mask_patch(mask, patch_size=8)`는 각 patch의 modal object ID를 택하고 동률이면 작은 ID를 택한다.
- 배경 ID는 0이다.
- metric은 pixel로 펼치지 않고 256 patch에서 직접 계산한다.

### Metric 정의

- `patch_fg_ari`: GT background patch를 제외하고, patch 쌍이 GT와 prediction에서 같은 cluster인지 비교하는 permutation-invariant ARI다. 예측 cluster ID를 GT ID에 미리 매칭할 필요가 없다.
- `patch_foreground_iou`: 모든 nonzero predicted label을 하나의 foreground로 합친 binary IoU다.
- `patch_matched_object_iou`: predicted foreground object와 GT foreground object 사이 IoU matrix를 만들고 Hungarian matching 후, GT object 수를 기준으로 평균한다. 없는/놓친 object도 패널티를 받는다.

### 금지 사항

- GT object 수로 predicted k를 정하면 안 된다.
- GT mask로 background group이나 threshold를 고르면 안 된다.
- reference test 점수로 classifier threshold를 고르면 안 된다.
- theta/PLV mask를 최종 prediction으로 제출하면 안 된다.

## 9. Slot Attention 기준

같은 reference IDs 1000–1319에 저장된 단일 checkpoint 기준은 다음과 같다.

- patch FG-ARI: `0.8901145722`
- patch foreground IoU: `0.2122510283`
- patch matched-object IoU: `0.2354868700`

주의: 이는 Slot Attention의 comparable 3-seed mean이 아니라 단일 checkpoint reference다. 최종 목표 선언에는 동일 seed 계약 또는 그 한계를 명확히 밝혀야 한다. validation의 평균 GT foreground object 수는 약 6.20, reference test는 약 6.04다.

## 10. 실험 기록 SW_0001–SW_0033

아래 점수 순서는 항상 `FG-ARI / foreground IoU / matched-object IoU`다.

- SW_0001: fixed-k spike synchrony classifier. `0.136286 / 0.271997 / 0.145604`. 기본 spatial-components보다 개선됐지만 Slot과 큰 차이.
- SW_0002: membrane-PLV 10 epoch pilot. `0.270323 / 0.235452 / 0.123274`. ARI 개선, object IoU 저하.
- SW_0003: matched 40 epoch membrane-PLV seed0. fixed-k 결과 `0.288924 / 0.207549 / 0.102696`. 이후 classifier 진단의 핵심 frozen checkpoint.
- SW_0004: graph-to-membrane teacher 10 epoch. `0.202940 / 0.245920 / 0.142077`. dendritic/membrane gradient 연결 확인.
- SW_0005: border background rule. 최고 foreground IoU `0.3220`, 그러나 ARI `0.0495`, object IoU `0.1386`; 미채택.
- SW_0006: peer connected-components spike readout. `0.077165 / 0.328720 / 0.173560`; 약 14.88 groups/image.
- SW_0007: SW_0006 foreground + fixed-k reclustering. 최고 `0.080966 / 0.328720 / 0.134053`; 개선 실패.
- SW_0008: graph-teacher core + component-product components. `0.195269 / 0.275711 / 0.238803`; 약 31 groups/image.
- SW_0009: SW_0008 frozen readout reference test. `0.190909 / 0.270928 / 0.229805`; Slot보다 foreground IoU만 높음. 예측 30.14 groups vs GT 6.04.
- SW_0010: component threshold 하향. group 수는 줄었지만 세 지표 모두 악화.
- SW_0011: graph-teacher 40 epoch. `0.179760 / 0.271468 / 0.295250`; object IoU 개선이나 77.03 groups/image로 심한 fragmentation.
- SW_0012: peer geodesic graph 진단. same/different edge ratio가 15.00→13.59로 하락하고 일부 negative distance; 미채택.
- SW_0013: adaptive spike-pattern slots. `0.247376 / 0.248332 / 0.204032`; groups 17.37. 아이디어는 유효하지만 과분할.
- SW_0014: adaptive slot background rule. IoU 일부 개선, ARI/object IoU 저하.
- SW_0015: gamma와 HDF5 ordering fingerprint가 정렬됨을 강하게 지지. RGB paired training이 가능함을 확인.
- SW_0016: RGB edge membrane loss weight 0.1, 40 epoch. `0.069476 / 0.331876 / 0.164980`; count 11.98로 감소했지만 grouping 악화. 미채택.
- SW_0017: adaptive slots input을 spike에서 continuous membrane으로 변경. `0.261507 / 0.246264 / 0.207637`; 소폭 ARI/object IoU 개선.
- SW_0018: low-synchrony foreground selection. `0.130456 / 0.291989 / 0.177966`; 미채택.
- SW_0019: spike/membrane correlation 혼합 pilot. 25% membrane에서 64-image 기준 `0.188753 / 0.268401 / 0.259725`; object IoU tradeoff.
- SW_0020: spike-source graph teacher 10 epoch. `0.122560 / 0.260356 / 0.287220`; object IoU 트랙, ARI 저하.
- SW_0021: SW_0003 checkpoint + adaptive slots. spike `0.313070 / 0.202942 / 0.107462`; membrane `0.357053 / 0.209870 / 0.113992`. membrane groups 5.98로 GT count 6.20에 매우 근접했지만 exact count 11.9%.
- SW_0022: membrane-border-prototype background. `0.361770 / 0.215081 / 0.119675`. 세 지표 소폭 동시 개선. predicted foreground 0.677 vs GT 0.217.
- SW_0023: oracle GT foreground 진단. foreground IoU 1.0, object IoU 0.385로 오르지만 FG-ARI는 0.357 그대로. 결론: background selection은 IoU 병목, grouping은 ARI 병목.
- SW_0024: membrane absolute-correlation spectral k=10. `0.393974 / 0.228821 / 0.150706`; 9 groups/image. 당시 ARI lead.
- SW_0025: spectral k=20. `0.348207 / 0.232265 / 0.228019`; object IoU 트랙.
- SW_0026: peer threshold sweep. SW_0011 threshold 0.10은 `0.098892 / 0.331330 / 0.180793`; foreground IoU 트랙이지만 ARI 악화.
- SW_0027: per-component positive-mean affinity k10. `0.386094 / 0.232912 / 0.157926`; aggregate k10보다 두 IoU는 조금 좋고 ARI는 조금 낮음.
- SW_0028: membrane spectral affinity에 spatial Gaussian sigma 1.5, k10. `0.494639 / 0.191070 / 0.170568`. 현재 FG-ARI-first lead였으나 foreground IoU 하락.
- SW_0029: sigma/k sweep. first64에서 sigma1.5/k10 유지. sigma1.25/k10은 object IoU 소폭 개선, ARI 하락.
- SW_0030: border-touching 전체 group을 background로 보내는 rule. foreground fraction은 GT에 가까워졌지만 ARI 0.530→0.261로 붕괴; 미채택.
- SW_0031: nonspatial positive membrane foreground gate, full320. `0.432699 / 0.205366 / 0.161077`; spatial control보다 IoU만 약간 높음.
- SW_0032: border-limited membrane veto width2. `0.467718 / 0.210417 / 0.174335`; 두 IoU tradeoff, ARI lead는 아님.
- SW_0033: 가장 중요한 최신 진단. spatial-only sigma1.5/k10 `0.491272 / 0.190837 / 0.178026`; actual membrane×spatial `0.494639 / 0.191070 / 0.170568`; membrane affinity를 위치 permutation해도 `0.481408 / 0.192980 / 0.165653`. GT foreground 인접 patch에서 raw membrane same-object vs different-object affinity mean 약 `0.89443 / 0.83773`, AUC `0.6365`; spatial-only AUC `0.6979`; membrane×spatial AUC `0.6968`. 결론: 최신 ARI gain 대부분은 공간 prior가 설명하며 learned membrane identity 신호는 약하다.

## 11. 현재 가장 중요한 과학적 결론

1. 초기 classifier는 심한 overfragmentation 때문에 object 수를 맞추지 못했다.
2. adaptive slots는 count를 GT 평균 근처까지 낮췄지만 FG-ARI가 충분하지 않았다.
3. background/foreground 판별은 IoU의 큰 병목이다.
4. foreground를 oracle로 완벽히 줘도 FG-ARI가 오르지 않았으므로 object grouping 자체가 별도 병목이다.
5. spectral classifier + spatial prior가 현재 FG-ARI를 크게 올렸지만, 이는 membrane이 object identity를 학습했다는 증거가 아니다.
6. 실제 membrane affinity는 같은 object와 다른 object를 어느 정도 구별하지만(AUC 약 0.6365), 공간 거리 자체보다 약하다.
7. 따라서 다음 연구는 `spatial-only control`을 항상 같이 두고, 학습 후 membrane/spike signal이 이 control을 명확히 초과하는지 확인해야 한다.

## 12. 다음 작업 방향

가장 먼저 새 대화에서 아래를 다시 확인한다.

1. 로컬 `patch_v2_sw`의 branch, HEAD, dirty files.
2. 서버 SSH mux 생존 여부와 `/Data0/kevinswk/patch_v2_sw`의 프로세스/PID/checkpoint/log.
3. `collaborative_test/STATUS.md`, `INDEX.md`, SW_0033 원문.
4. `git fetch origin patch_v2` 후 동료의 새 실험 결과가 실제로 추가됐는지.

그다음 무작정 SW_0034 training부터 하지 말고 다음 기준점을 만든다.

- 동일 이미지에서 theta PLV affinity, gating waveform affinity, dendritic/h-wave affinity, membrane affinity, spike affinity를 같은 GT foreground neighbor-pair 또는 all-pair 진단으로 비교한다.
- 각 단계에서 same-object/different-object 평균, AUC, spatial-only 대비 incremental gain을 기록한다.
- gradient connectivity와 parameter group별 norm도 함께 확인한다.
- 어느 단계에서 object signal이 줄어드는지 정한 뒤 loss를 그 단계에 연결한다.
- classifier는 현재 spatial-only와 membrane×spatial spectral k10을 control로 유지한다.

### SW_0034 상태

`collaborative_test/SW_0034_membrane_edge_training`이라는 빈 디렉터리 이름이 한 번 준비됐지만, 실험 문서·코드·학습은 시작되지 않았다. 빈 디렉터리는 Git에 추적되지 않는다. 작은 RGB edge membrane weight를 matched SW_0003 설정에서 시험하는 아이디어가 있었으나, SW_0016 weight 0.1의 실패가 있으므로 먼저 단계별 signal diagnostic을 수행하는 편이 현재 사용자의 방향과 더 잘 맞는다.

## 13. 동료 브랜치에서 얻은 정보

- 동료 결과는 다른 render/split/reference를 사용할 수 있어 절대 점수를 우리 결과와 직접 합치면 안 된다.
- 마지막으로 알려진 peer best PV2_0004 3-seed test는 대략 `0.5105 / 0.4967 / 0.3689`였지만 평가 계약 차이를 반드시 붙여야 한다.
- peer PV2_0005 oracle graph 진단은 downstream readout/spiking이 병목일 가능성을 제시했다.
- peer에서 spatial decay 0.35가 도움이 되고 geodesic 5가 악화된 기록이 있었다.
- 최신 `b99fac0`은 상태/재개 문서 중심이며, queued experiment가 실제 완료됐다는 뜻이 아니다.
- 매 cycle `git fetch origin patch_v2` 후 `collaborative_test`의 새 commit과 실제 결과 파일을 확인한다.

## 14. 서버 접근과 안전 규칙

사용자가 먼저 PowerShell에서 WSL을 열고 `ssh frontier`로 password-authenticated master connection을 만든다. 에이전트는 사용자의 열린 multiplexed session을 통해 명령만 보낸다. 비밀번호를 요청하거나 저장하지 않는다.

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'ssh -O check frontier'
wsl.exe -d Ubuntu-24.04 -- bash -lc 'ssh -o BatchMode=yes frontier "<server command>"'
```

서버 conda:

```bash
source /Data0/kevinswk/miniforge3/etc/profile.d/conda.sh
conda activate /Data0/kevinswk/envs/snn
```

- GPU: TITAN RTX 24GB ×4.
- 실행 직전 `nvidia-smi`로 memory와 utilization을 모두 보고 가장 비어 있는 GPU 하나만 사용한다.
- 다른 사용자 프로세스를 kill하거나 바꾸지 않는다.
- 긴 실행은 `nohup`과 명시적 log/checkpoint 경로로 session과 분리한다.
- 서버 `/Data0/kevinswk/patch_v2_sw`는 Git 저장소가 아닐 수 있으므로 로컬 GitHub branch가 코드의 권위 있는 source다.
- 서버 root filesystem이 꽉 찬 적이 있으므로 tmp/cache는 `/Data0/kevinswk/.../tmp` 같은 Data0 경로를 사용한다.

## 15. collaborative_test 관리 규칙

- 번호는 `SW_0001`, `SW_0002`처럼 증가시키고 재사용하지 않는다.
- 상태: planned, running, completed, failed, interrupted.
- 각 실험은 최소한 README, config, commands, changes, provenance, metrics, diagnostics/artifact 경로를 기록한다.
- 큰 checkpoint나 전체 raw log는 Git에 무조건 넣지 않는다. 핵심 결과, 작은 JSON/CSV, 재현 명령, hash와 서버 경로를 남긴다.
- validation-selected setting만 reference test로 보낸다.
- 실패한 실험도 원인과 배운 점을 기록한다.
- improvement가 확인되면 문서와 필요한 코드를 commit/push한다.
- 동료 브랜치는 읽고 분석하되 force-push하거나 직접 수정하지 않는다.

## 16. 새 대화에서 즉시 사용할 작업 순서

1. 이 문서를 읽고 현재 연구 목적을 한 문단으로 재진술한다.
2. `patch_v2_sw` 실제 Git 상태와 최신 `collaborative_test/STATUS.md`를 확인한다.
3. 서버 mux와 live process를 read-only로 확인한다.
4. 동료 `patch_v2`를 fetch하고 마지막 검토 이후 새 결과만 분석한다.
5. 이전 goal을 자동으로 이어받았다고 주장하지 말고, 다음 목표를 새 goal로 생성한다:
   - “고정 patch 평가 계약에서 spike/membrane-derived masks의 3-seed mean이 comparable Slot Attention mean을 세 지표 모두에서 초과하도록, 진단 기반으로 patch_v2_sw를 반복 개선하고 모든 실험을 collaborative_test에 기록·공유한다.”
6. 먼저 stage-wise object-signal diagnostic을 설계하고, 결과를 바탕으로 다음 controlled training experiment를 선택한다.
7. 사용자에게 routine epoch가 아니라 major finding, 코드 변경, 의사 결정이 필요한 경우만 보고한다.

## 17. 반드시 피해야 할 오해

- 현재 최고 FG-ARI가 0.4946이라고 해서 learned binding이 강하다고 말하면 안 된다. SW_0033에서 spatial-only가 0.4913이었다.
- object count가 평균적으로 GT와 비슷하다고 segmentation이 맞는 것은 아니다. SW_0021은 count는 가까웠지만 exact count와 ARI가 낮았다.
- foreground IoU와 FG-ARI는 다른 병목이다. foreground를 맞춰도 grouping ARI는 그대로일 수 있다.
- PLV 성능은 최종 object mask 성능이 아니다.
- 동료 branch의 더 높은 점수를 우리 split의 Slot reference와 직접 비교하면 안 된다.
- reference test를 반복해서 threshold selection에 쓰면 안 된다.
- `spike`라는 함수 인자 이름이 실제 binary spike인지, `loss_signal`로 선택한 sigmoid membrane activity인지 호출부에서 확인해야 한다.

## 18. 새 대화 시작용 한 문장 요약

현재 `patch_v2_sw`는 learned membrane/spike 기반 object grouping을 목표로 하지만, 최신 SW_0033에서 최고 ARI readout의 대부분이 spatial prior로 설명됨이 확인되었으므로, 다음 단계는 theta→gating→dendritic→membrane→spike 각 단계의 GT 분리 신호와 gradient를 정량화하고 신호가 사라지는 지점에 loss를 연결한 뒤, spatial-only control을 넘는 spike/membrane-derived mask를 3-seed 고정 평가로 검증하는 것이다.
