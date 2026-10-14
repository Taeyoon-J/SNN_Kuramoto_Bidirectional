# SW161 prospective — actual-spike partition에 직접 object-mask supervision

2026-10-14. 설계만 등록한다. SW160 진행 코드/queue는 변경하지 않는다. 실행은 SW160 final 결과 기록 후 별도 구현 검토를 거친다. 이 문서는 **supervised track**이며 unsupervised 성능 주장과 분리한다.

## 실제 코드의 bottleneck

`spike_classifier.py::spike_synchrony_affinity`는 각 component의 시간 중심화 Pearson을 양수 clamp한 뒤 곱한다. 동일한 네 correlation이라면 기존 Q=.5 edge를 만들려면 각 correlation이 약 .8409 이상이어야 한다. 서로 다른 component에서는 하나의 음수 correlation이 product를0으로 만든다. 이는 source의 강점과 함께 고정된 표현 제한이다.

`SW0155/objective.py`의 positive signed-correlation 항은 막힌 positive pair credit을 보완하지만, 최종 `spike_synchrony_components`는 Q>=.5의 **transitive connected components**를 만든다. 한 bridge가 두 object를 합칠 수 있고, 가장 큰 component를 background로 제거하며 작은 그룹은 버린다. Loss는 이런 전체 partition/background decision을 직접 학습하지 않는다. 그렇다고 이 mismatch가 전체 실패의 원인으로 입증된 것은 아니다.

SW133은 current QCC가 정한 H/K에 soft credit을 주므로 K=1과 기존 group 구조의 제한이 있다. SW150 `seed_selection.py`도 current QCC region count에서 FG seed 수를 결정한다. 반면 SW135 binder는 full `[B,4,1024,512]` spikes를 읽는 imagewise three-iteration Slot Attention이며 K는 QCC에 묶이지 않는다. 이전 RGB decoder/completion supervision은 object-instance 정답을 구분하지 못했다. **그 RGB objective를 재사용하지 않고**, 최종 object assignment에 TRAIN mask supervision을 직접 적용한다.

## 하나의 matched experiment: 동일 learned partition, 다른 supervision

source97 seed1의 native32 foundation에 SW143-T operator를 설치하고 encoder/graph/core/integration를 모두 freeze한다. SW159 threshold나 SW158 emitter는 사용하지 않는다. SourceT의 원래 actual-spike QCC는 immutable reference/secondary baseline이다.

두 arm 모두 `SW0135/binder.py::NativeSpikeSlotBinder`의 full-time input projection,64-dimensional latent,three shared GRU iterations를 같은 seed로 사용한다. Input은 **actual emitted component spikes의 settled512뿐**이다. RGB/gamma/theta/membrane/XY를 head에 넣지 않는다. RGB decoder는 생성하지 않는다. Mean-time pooling은 사용하지 않는다.

`NativeSpikeSlotBinder`의 constructor는 module 상수 `SLOTS=11`을 사용하고 `initial_noise[1,11,64]`를 저장한다. `forward`는 actual spike tensor만 받으며 QCC K/seed/H를 받지 않는다. 따라서 QCC가1group인 이미지에서도11개 assignment를 생성할 수 있다. 동일 constructor seed161001로 양쪽을 초기화하고 별도 BG-special slot/initialization은 추가하지 않는다.

**11개 slot은 모두 exchangeable이며 특정 slot을 inference BG로 지정하지 않는다.** TRAIN candidate만 아래 Hungarian matching을 수행한다. Inference에서는 GT count/matching 없이 P.argmax를 계산하고, 양쪽 모두 가장 많은 patch가 hard-assigned된 nonempty group을 background로 정한다. 동률이면 해당 group의 첫 row-major patch index가 작은 것을 택한다. 이 group과 size8 미만 group을 label0으로 remap하고, 나머지는 첫 patch 순서로 canonical positive IDs를 부여한다. Empty slots는 무시한다. 고정 occupancy/object count는 없다. SourceQCC reference는 원래 코드 그대로다. 이는 **새 learned readout**이며 IDs/modal32/세 metrics 및 largest-background 정책이라는 고정 평가 계약은 변경하지 않는다. Slot reference의 이미 등록된 pixel prediction/BG 정책도 다시 바꾸지 않는다.

- **control — pair_supervised_P:** coassignment `C_ij=sum_k P_ik P_jk`에 SW155의 unordered sameFG/differentFG/FGBG/BGBG class-balanced relation MSE(각 image의 nonempty class means를 평균)를 적용한다. Target은 same-instance1/different-instance0. Signed-rho 보조 항은 P에 의미가 없으므로 추가하지 않는다.
- **candidate — mask_supervised_P:** same P에 instance Hungarian CE + Dice를 적용한다. Matching cost는 각 GT instance patch의 mean negative log-probability + soft Dice cost다. **TRAIN에서 실제 존재하는 BG를 포함한 모든 GT labels를11개 slot에 Hungarian-match**하고 unmatched slot의 target은 empty로 둔다. Named BG0 supervision을 후보에만 추가하지 않는다. TRAIN의 total distinct labels가 capacity11을 넘으면 숨기거나 잘라내지 않고 fail closed한다. TRAIN matching 결과/GT BG 선택을 inference에 전달하지 않는다.

Candidate CE는 실제 존재하는 GT labels 각각의 patch CE를 먼저 평균한 다음 labels를 평균하여 background만으로 좋은 loss를 얻기 어렵게 한다. Dice는 matching된/empty11개 target channels에 `1-(2*sum(P*Y)+1)/(sum(P)+sum(Y)+1)`을 적용해 평균한다. Total=CE+Dice, 두 계수1. Control의 P·Pᵀ loss와 candidate의 matching loss 모두 slot permutation에 invariant이며, 양쪽 inference BG 규칙은 같다. 별도 RGB loss/density/teacher/source anchor를 추가하지 않는다.

이 paired 비교는 동일 head에 **object-level mask objective가 relation surrogate보다 더 유용한가**를 시험한다. Head 개선과 core 개선을 한 번에 섞지 않는다. Object partition은 일반적으로 unsupervised consistency/RGB만으로 식별 가능하지 않지만 여기서는 허용된 TRAIN masks가 instance identity를 제공한다.

## 비용을 제한하는 고정 pilot

source 순서의 첫512 unique TRAIN IDs를 사전에 고정한다. 이 subset에 대해 frozen sourceT B4/T1024/settle512의 실제 spikes를 **원래 float32 byte 그대로** 한 번 cache한다(약4GiB). Cache에는 정확한 dtype/shape/IDs/order/source/encoder/stats/operator/source-file hashes와 source no-mutation receipt가 필요하다. 첫 실제 B4를 live sourceT와 exact replay로 확인한다. Float16으로 저장 후 float32로 cast한 자료는 exact-source cache로 인정하지 않는다. 기존 RGB/16-grid/다른 graph operator cache를 이 cache로 간주하지 않는다.

Classifier 학습은 same logicalB16,16 complete passes =512 updates, Adam LR3e-4,global clip1. 두 arm의 order/shuffles/initial weights는 같다. Decoder warmup이 없고 core rollout 재계산 없이 cached actual spikes로 학습하므로 새 head에 한 pass만 주는 것보다 학습 시간을 확보하면서 비용을 제한한다. Final update512만 평가하고 validation checkpoint selection/early stopping을 하지 않는다. 전체 supervised QCC reference가 가진 pretrained70k lineage와 cache에 쓰인 추가 image exposures를 공개한다.

## Admission / endpoint / close

Actual cached `[B,4,1024,512]`로 forward/backward가 실행되고 head parameter가 변경되며 source checkpoint/encoder/statistics가 불변이어야 한다. P가 finite,row-sum1이고 supervised loss가 finite인지 확인한다. Hungarian 전체-slot permutation에 따른 loss invariance,common largest-background/size8 remap,empty slot/no GT-count inference를 CPU 회귀로 검사한다. 초기 FG나 RGB shuffle score로 architecture를 reject하지 않는다.

실제 fixed320 RGB로 frozen sourceT T1024/512를 실행하고 trained head로 native32 labels를 만든다. Cached TRAIN traces를 validation에 사용하지 않는다. 양쪽P/sourceT/Slot references의 fixed320 predictions를 모두 고정한 후 modal4px GT와 FG-ARI/FGIoU/objectIoU를 계산한다. SourceT actual-QCC도 별도로 보고한다. Candidate primary를 QCC로 이름만 바꿔 보고하지 않는다.

Candidate FG가 sourceT와 matched control 모두보다 개선되고 두 paired-image CI lower bounds가 양수일 때만 seed0/2 replication을 고려한다. .87 target/세 IoU 목표에 대한 절대 차이를 보고한다. 실패하면 learning-rate/K/threshold/decoder sweep 없이 이 pilot을 닫되512 unique images의 학습이 absolute supervised ceiling을 증명했다고 주장하지 않는다.

유망하면 다음 별도 registration에서 trained head를 포함해 encoder/graph/core joint 학습 및 fresh native32 70k/three-seed supervised 확인을 한다. Positive data scaling은 fixed-update와 fixed-epoch 설계를 분리한다. TRAIN mask를 쓴 성공은 unsupervised Slot superiority를 증명하지 않는다.

## Paper/gating 한계

현재 head input은 actual spikes지만 saturated source에서는 그것이 거의 shared delayed-gate 파형이다. Head 정확도만으로 membrane/event/history 기여를 주장할 수 없다. 유망 모델에서 source-preserving/frozen signal substitution과 실제 dynamics intervention 및 matched retrained ablation을 별도 수행해야 한다. 이 실험은 .87을 보장하지 않으며 모든 gate 기여·unsupervised70k/scaling 목표는 유지된 미완료 과제다.
