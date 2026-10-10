# 실제 spike 관계 보존 + cross-view 목적함수: 조건부 설계

2026-10-10, Sol6.1 검토. 실행 등록이나 실험번호 할당이 아니다. SW0139 cross_view 및 SW0140의 all3/full320 결과를 먼저 완료한다. 현 실험·guard·queue는 변경하지 않는다.

## 근거와 질문

완료된 partition 진단에서 context의 false-merge 증가 .039268 중 .037939(96.6%)는 predicted BG column이다. Missed FG는 .130703→.256129, false-split 증가는 .000868이다. 따라서 전경을 BG에 흡수하는 관계 drift가 주요 관찰 패턴이다. 이것이 density prior나 clipping의 단독 인과효과라는 뜻은 아니다.

다음 질문은 **기존 관계를 훼손하는 통계적 목적 최적화 대신, 원본 spike 관계를 보존하며 view 안정성을 학습할 수 있는가**이다. Source는 정답 teacher가 아니다. SW0128의 실패한 binary pseudo-instance precision을 우회하거나 재해석하지 않는다.

## 동일 모델의 최소 대조

동일 seed1 SW0137 native32/source97 strict-mapped core와 registered encoder/statistics에서 시작한다. Encoder와 graph만 trainable; native dynamics와 12 scalars는 고정한다. Context/decoder/classifier 추가 없이 실제 [B,4,1024,1024] spike trace를 사용한다.

두 새 arm은 모두 old phase+5Q 통계적 목적을 **완전히 제거**한다.

- C-only: `L=λ_C C`.
- Source-preserving: `L=λ_C C+λ_A A`.

완료된 SW0139 old+λ_C C를 같은 seed/order/optimizer/augmentation의 기존 reference로 별도 비교한다. C-only↔C+A는 anchor의 추가효과, old+C↔C-only는 phase 및 Q priors 전체 교체의 효과를 구분한다. 이 비교로 balance/density 하나의 원인이나 phase prior만의 효과를 주장할 수 없다. 신규 두 arm을 old+C 하나와만 비교하면 삭제와 anchor 추가가 섞이므로 충분한 인과 대조가 아니다.

C는 실제 SW0139 helper 그대로: settled512의 raw 4×512 spike를 flatten/L2 normalize, 같은 patch의 밝기·contrast 변형 view를 positive로 하고 B4 내 다른 image patch32개만 negative, symmetric InfoNCE τ=.1. Augmentation/negative RNG nonce도 원래 fixed order에 결박한다. 같은 image의 다른 object를 negative라고 가정하지 않는다. 단, 이 C 자체가 instance 감독을 제공하는 것은 아니다.

## Soft numerical anchor와 teacher 비용

원본 frozen same-seed encoder/graph/core의 **canonical RGB** rollout에서 production `spike_synchrony_affinity(..., settle=512, affinity_mode='spike')`로 `Q_T`를 만든다. 각 component correlation의 positive clamp와 product를 정확히 유지한다. Teacher에는 GT, threshold, Hungarian label, H/group/BG 판정이 없다.

학생도 같은 canonical RGB에서 `Q_S`를 계산하고,

`A = mean_images [ sum_{i≠j}(Q_S[i,j]-stopgrad(Q_T[i,j]))² / (1024×1023) ]`.

대칭 양방향을 그대로 평균하고 diagonal은 제외한다. Augmented view는 C에만 사용한다. Uniform pair 평균은 source의 희소한 FG 관계를 약하게 반영할 수 있고, 작은 MSE가 .5 CC의 bridge 안정성을 보장하지도 않는다. 따라서 A 감소나 teacher 충실도만으로 object 개선을 선언하지 않는다.

권장 방식은 고정 4096 canonical TRAIN Q를 **float32 전체 matrix**로 한 번 생성하는 것이다: 4096×1024²×4 bytes = 16 GiB payload. Float16/quantization/binary confidence filtering은 사용하지 않는다. 비교할 두 새 arm이 같은 immutable cache를 재사용한다. 생성은 no-grad full1024/512, 실제 B4 및 동일 microbatch grouping으로 수행한다. Native recurrence의 batch-shape 수치 차이를 없애기 위해 B1로 만든 teacher를 B4 학생 reference로 조용히 대체하지 않는다.

두 번째 teacher rollout을 매 update 반복하는 방식은 storage는 줄지만 두 view 학생에 canonical teacher 비용까지 계속 더한다. Shared teacher cache는 두 pilot에서 그 비용을 한 번만 지불하므로 우선한다. Source0 보정용 첫64 teacher Q는 256 MiB이며 별도 source0 provenance를 유지한다. 실제 생성·I/O·학생 two-view update 시간을 먼저 측정하고 비용을 보고한다; 기존 one-view 약20초/logical update를 two-view 비용으로 가정하지 않는다.

Cache ledger는 source core/encoder/statistics/graph 설정 및 helper SHA, RGB bytes/preprocessing, ordered image IDs와 pool rows, B4 grouping, full1024/512 horizon, matrix dtype/shape와 chunk hashes에 결박한다. 각 image의 Q row/column은 동일 native32 row-major patch ordering이다. Teacher/source는 optimizer를 갖거나 학생 checkpoint에서 갱신하지 않는다. 첫 actual batch의 cache↔동일 teacher 재계산과 source 학생 Q parity를 확인한다.

## 정확한 TRAIN-only weight 규칙

`λ_C`는 SW0139 source0 첫4 logical TRAIN batches의 고정 calibration 값과 provenance를 재사용한다. 두 새 arm에서 동일하다. 이는 old gradient를 기준으로 정해진 기존 scale이라는 한계를 기록하고, validation에서 다시 맞추지 않는다.

동일 source 초기에는 A=0, ∇A=0이므로 초기 gradient ratio로 λ_A를 만들면 안 된다. 다음 고정 disposable 절차를 등록한다.

1. Fresh source0와 fresh Adam, encoder3e-6/graph3e-5, logicalB16/B4×4, unique-union clip1로 **첫 TRAIN B16의 λ_C C만 1회 update**한다. 실제 objective와 동일 RNG를 사용한다.
2. 이 고정 probe 상태에서 첫4 logical TRAIN batches를 재평가한다. 각 batch의 B4 gradient vectors를 1/4씩 먼저 누적한 뒤 encoder+graph 전체 norm을 계산한다.
3. `λ_A = median_batches( ||∇(λ_C C)||₂ / ||∇A||₂ )`로 설정한다. 이는 한 source0 C-induced displacement에서 두 목적의 credit scale을 같게 하는 단일 규칙이다. 분모의 zero/nonfinite는 calibration 실패로 기록하며 epsilon으로 통과시키거나 다른 step/seed로 재보정하지 않는다.
4. Probe weights와 Adam은 폐기한다. 두 실제 arm은 원본 source와 fresh Adam에서 시작하고 λ_A는 다른 seed에도 그대로 동결한다. 첫 update에서 anchor gradient가 0인 것은 정상이다. 보정 원본/상태 hash, 각 목적 family norm과 cosine, 최종 λ를 보존한다.

보정 이후 두 목적 gradient는 parameter union에서 합산하여 한 번 clip/Adam한다. λ를 family별로 추가 조정하거나 event occupancy에 맞추지 않는다. 단계2의 credit 검사와 source/cache fidelity가 충분한 preflight이며, 새로운 GT precision guard를 만들지 않는다.

## 예산·종료·논문 한계

Pilot은 기존과 동일 seed1, ordered4096 unique images, 256 logicalB16/B4×4, full1024/settle512/live64, fresh Adam, encoder3e-6/graph3e-5, clip1이다. 두 image views는 8192 view exposures로 공개한다. Train은 GT-free이고 endpoint는 immutable native32 modal4px GT와 actual-spike QCC fixed320이다. 저장된 final checkpoint 하나만 평가한다.

확장을 고려하려면 C+A의 FG가 source 및 C-only를 모두 넘어 common-image paired95% lower bound가 양수여야 한다. 완료 old+C와의 비교 및 세 IoU/ARI 지표도 함께 보고한다. 실패하면 이 고정 recipe를 닫고 λ/teacher threshold/precision/seed 변경으로 반복하지 않는다. 단일 seed 결과로 all3 Slot superiority를 주장하지 않는다.

Source를 복제하는 것만으로 모든 gate의 usefulness나 scaling이 해결되지 않는다. 특히 본 설계는 membrane/event parameters를 학습하지 않는다. SW0140 frozen intervention은 필요성 경고이고 retrained gate 대조와 다르다. 유익한 representation 학습이 확인될 때만 별도 event-credit 대조의 우선순위를 검토한다. 최종 all3 comparable Slot32 세 지표 및 increasing-unique-data 개선은 별도 목표다. 70k lineage continuation을 fresh-data scaling으로 부르지 않으며, 비용과 three-seed 성능을 확인하기 전 full70k 확장하지 않는다. Reserve90640–90959는 읽지 않는다.
