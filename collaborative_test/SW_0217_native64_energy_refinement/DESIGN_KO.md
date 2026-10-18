# SW0217 prospective: native signed-energy split and remerge

실행 전 고정한 구현 명세다.216까지의 merge-only classifier는
개선되지 않았다. 현재 baseline 각 group은 이미4-connected이므로 단순
connected-component 분리는 반복하지 않는다. Native greedy agglomeration의
초기 잘못된 merge를 실제 동일 energy에서 되돌릴 수 있는지 시험한다.

## 고정 source와 평가 범위

- 실제210 기준점의 독립 source0/1/2를 사용한다.0 warm192 model SHA
  2dd2b0d64602b85d4082bcaddcd21dfe8c60d34af1ea78a2ed29bf03c8ba4b6e;
  1 warm205 e1e16281612b2171b6822e0ca70dee051f095ad7ebc9239a9675b8e8d9b391e0;
  2 warm205 dff381ebcd5280a2a7bf7b99ef8ee3b5af30e91f3384962ef47c6d89b7da53d0.
- Seed0 factory: SW0193.prepare_inputs_cpu의 found에 candidate_state를 실제
  복원한다. Seed1/2: SW0210.prepare_cpu(seed)를 그대로 재사용한다.
- 같은16 offsets0,2,..30/IDs1320..1350, canonical128 RGB/no crop/native64,
  inference RNG193000+offset,1024 steps/settle512/live_tail0. 학습 update0.
- 원래 baseline mask를 모든 image에서 bitwise 재현해야 후보를 허용한다.
  Core/encoder/graph/gating은 유지하며 실제 model SHA를 전후 확인한다.

## 에너지와 새 optimizer

같은 actual Q/A/gain에서 SW0162.signed_pair_energy를 그대로 호출한다.
NATIVE_NODES 상수나 graph normalization을 이번에 바꾸지 않는다.268-channel
anchor cache로 full energy를 복구했다고 하지 않는다. 새 core forward가 필요하다.

기존 group마다 E_block의 signed Laplacian diag(E.sum(1))-E를 계산한다.
최소 eigenvector의 zero cut과 sorted-vector cut 중 cross-energy가 최소인
prefix cut을 고정 후보로 삼는다. 각 side를 공간4-connected pieces로
나눈 뒤, 실제 전체 within-group energy가 증가하는 후보만 수락한다.
동일성/tie는 row-major 순서로 처리한다. 허용오차는 float64 machine epsilon과
실제 abs-energy 합에서만 정하고 GT/metric으로 고르지 않는다. 원래 group당
한 번만 proposal한다. 추가 recursive/eigenvector/threshold sweep은 없다.

Secondary split-only와 primary split-then-remerge를 함께 등록한다. Remerge는
split된4-connected groups에서 시작해 기존과 같은 full cross-energy 합이
양수인 인접 group을 최대 gain/canonical tie 순으로 합친다. 원래 source FG는
모두 보존하며 singleton도 버리지 않는다. BG 재선택/min-size/count cap 없음.
모든 group4-connectivity, FG exact equality, energy nondecrease를 검증한다.

## 검증과 저장

Operator CPU tests: synthetic negative cross-energy split/positive clique identity,
connectivity after eigen cut, exact foreground, label-ID permutation invariance,
deterministic tie, split/remerge energy monotonicity, empty/singleton, invalid inputs.
Native source/CUDA 전 실제 CPU factory admission과 source reference hashes를 확인한다.

GPU0..3 exclusive lease/per-image checkpoint를 사용하고 외부 owner에게 양보한다.
각 row는 baseline/split/refined masks와 full-N 계산 후 자른 실제 Q_F/A_F,
F indices/gain을 dtype 손실 없이 server-private NPZ에 저장한다. Full Q/A와
energy hashes도 기록한다. 정규화를 F 크기로 바꾸거나 foreground oscillator를
다시 실행하지 않는다. 저장 전에 실제 free disk와48행 최악 크기를 확인한다.
외부 auditor는 원래 operator 상수로 Q_F/A_F에서 E_F를 재계산하고 실제
mask energy와 operator replay를 검증할 수 있어야 한다. Local/GitHub에는
metadata만 내보내며 Q/A/mask arrays/checkpoint/RGB를 공개하지 않는다.

세 seed의 모든 endpoint를 고정한 뒤 GT를 읽고 세 지표·count·energy gain·
accepted split 수·paired CI를 보고한다. 같은16의 세 seed 평균이므로 full320
.90 또는70k 완료 주장이 아니다. No split/energy gain이면 후보를 찾지 못한
것이며 global optimum/모든 energy cut 불가능을 증명했다고 하지 않는다.
Energy가 좋아져도 mask metric이 좋아진다고 가정하지 않는다.

확대 후보는 primary의 세 seed mini 평균 FG delta>=.01, paired95% CI lower>0,
ObjectIoU 평균 감소 없음으로 판정한다. FG-only 후보도 별도 보존하지만 작은
양의 point estimate만으로 전체320/70k를 확장하지 않는다.

개선이 크고 재현되면 전체320에서만 최종 비교한다. Gate 필요성이나 data
scaling은 이 readout-only 시험으로 입증되지 않는다. 기존 .848615 reference와
216의 실패 판정은 유지한다.
