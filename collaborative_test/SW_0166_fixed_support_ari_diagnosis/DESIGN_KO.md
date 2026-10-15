# SW0166 prospective — fixed-support FG-ARI bound diagnosis

2026-10-14 preregistration. SW0165 seed0/2 score를 열기 전에 고정한다. SW0164 seed1과 SW0165 seed0/2의 prediction이 모두 완료·검증되면 **SW0165 gate 통과 여부와 무관하게 세 seed 전부**를 진단한다. 성공 seed만 고르지 않는다. 이 시험은 저장된 prediction의 GT-last 사후 진단이며 재학습, 재예측, recipe 변경 또는 자동 promotion이 아니다.

## 고정 입력과 GT 순서

- fixed validation IDs `1320..1639`, native modal32 GT, production foreground definition/FG-ARI를 그대로 사용한다.
- seed1은 SW0164의 immutable `foreground_induced`와 fresh SourceT control, seed0/2는 SW0165의 대응 prediction/control만 사용한다. 각 prediction/protocol hash와 fixed-support exactness를 GT open 전에 검증한다.
- 세 seed의 candidate와 control 320개가 모두 freeze·검증된 뒤에만 canonical validation GT를 한 번 연다. reserve `90640..90959`, TRAIN GT, 다른 validation image를 읽지 않는다.
- [`../SW_0165_foreground_partition_replication/FIXED_SUPPORT_ARI_BOUND_20261014.md`](../SW_0165_foreground_partition_replication/FIXED_SUPPORT_ARI_BOUND_20261014.md)의 metric convention과 증명을 사용한다. unlimited/disconnected positive group relaxation의 최대값이며, 4-connected 실제 solver feasible set의 인증 상한이다. 실제 solver가 disconnected group을 생성할 수 있다고 주장하지 않는다.

## 이미지별 고정 측정

Production FG-ARI와 똑같이 GT foreground patch만 pair 모집단으로 삼는다. `n<2`이면 production처럼 NaN이며 모든 평균/bootstrap에서 동일한 valid-image mask와 valid count를 명시한다.

각 seed/image에 다음을 저장한다.

1. `actual_fg_ari`: frozen candidate의 production FG-ARI.
2. `fixed_support_ari_upper_bound`: candidate/SourceT의 zero/nonzero membership을 고정하고, covered GT foreground를 GT object별 unlimited positive label로 나눈 인증 상한.
3. support counts: GT-foreground `covered_patch_count`, `missed_patch_count`, covered/missed가 하나 이상 있는 `covered_gt_object_count`, `missed_gt_object_count`, 그리고 두 support에 걸친 `cross_support_gt_object_count`.
4. actual positive partition의 **원시 pair 수**:
   - `covered_merge_pair_count`: covered이면서 서로 다른 GT object인데 같은 positive candidate group인 pair.
   - `covered_split_pair_count`: covered이면서 같은 GT object인데 서로 다른 positive candidate group인 pair.
   - `forced_bg_merge_pair_count`: 둘 다 missed이고 서로 다른 GT object라서 fixed label 0에서 합쳐지는 pair.
   - `forced_cross_support_split_pair_count`: 같은 GT object에서 한 patch는 missed, 다른 patch는 covered라서 fixed zero/nonzero membership상 분리되는 pair.
   각 count와 함께 동일 이미지의 total GT-foreground pair 수, same-GT pair 수, different-GT pair 수도 저장한다. 서로 denominator가 다른 rate끼리 크기를 비교해 어느 오류가 지배적이라고 부르지 않는다.
5. 등록된 순서의 counterfactual FG-ARI:
   - `after_split_mixed_positive`: actual positive cluster를 GT object별로 쪼개 covered merge pair를 제거하되, fixed label-0 membership과 같은-GT positive split은 유지한다.
   - 그 다음 같은 GT object의 positive group을 합친 값이 `fixed_support_ari_upper_bound`다.
   - `merge_correction_delta = after_split_mixed_positive - actual`, `split_correction_delta = upper_bound - after_split_mixed_positive`, 두 값의 합이 `upper_bound - actual`과 수치 허용오차 안에서 같아야 한다. 이 두 delta는 등록된 순서에 의존하는 설명량이며 독립적 인과효과로 해석하지 않는다.
6. `support_idealized_attachment_fg_ari`: covered candidate group은 유지하고, missed GT-foreground patch를 같은 GT object와 overlap이 가장 큰 기존 positive group에 붙이며 overlap이 없으면 object별 새 positive label을 준다. 이는 GT-assisted 참고값이다. support만의 효과, 실제 predictor 성능, 달성 가능한 unsupervised 성능으로 주장하지 않는다.

## 집계와 불확실성

seed별 320-image mean/valid count를 먼저 보고한다. 주 결정표는 같은 image index에서 seed0/1/2 값을 평균한 320-vector를 사용한다. paired bootstrap은 image index를 공동 resample하고 `10,000`회, RNG seed `166`으로 고정한다. 다음 contrast의 mean과 paired 95% CI를 보고한다.

- certified upper bound minus actual candidate
- certified upper bound minus `.87`
- certified upper bound minus seed-matched Slot70k FG-ARI (세 seed 평균 vector)
- actual candidate minus seed-matched Slot70k FG-ARI
- 두 순차 correction delta

`.87`은 각 image에 상수 `.87`을 둔 contrast다. seed별 값도 모두 공개하지만 유리한 seed만 골라 결론내리지 않는다. Pair count는 image별 raw count의 합과 평균을 보고하며, 다른 denominator의 normalized rate 차이를 오류 기여도의 순위로 사용하지 않는다. Bootstrap CI는 fixed320 밖의 image 모집단에 대한 불확실성 근거로만 사용하고, 이미 관측된 fixed320 board의 결정론적 가능 여부를 바꾸지 않는다.

## 제한된 판정

1. **Fixed320 deterministic flags를 두 개 따로 계산한다.** 세-seed mean certified bound가 `.87`보다 작으면 `fixed320_cannot_reach_0p87=true`; `.87`과 같으면 등록된 `>=.87` 조건은 가능하므로 false다. Bound가 seed-matched Slot70k 세-seed mean보다 작거나 같으면 `fixed320_cannot_strictly_beat_slot=true`; equality는 strict win이 아니므로 true다. 두 목표를 모두 요구하는 결합 FG 목표는 둘 중 하나라도 true이면 현재 fixed supports로 불가능하다. 둘 다 false일 때만 relaxation 안에서 결합 목표가 가능하다고 기록한다.
2. 위 fixed320 flag는 exact board mean 비교로 결정하며 bootstrap CI 때문에 `inconclusive`로 바꾸지 않는다. 별도 **generalization evidence**에서 각 bound-minus-target paired CI가 전부 양수면 `positive`, 전부 음수면 `negative`, 0을 가로지르면 `inconclusive`로 각각 기록한다. `.87`과 Slot contrast를 서로 합쳐 하나의 CI로 만들지 않는다.
3. fixed320 certified bound가 해당 목표를 허용하지만 actual이 못 넘으면, fixed support 안에 grouping headroom이 있다고 판정한다. 등록된 순차 delta와 raw pair counts는 다음 설계에서 merge 또는 split 중 어느 오류를 직접 다룰지 묻는 근거지만, 이 진단만으로 loss/threshold/group count를 선택하지 않는다.
4. fixed320 불가능 flag는 foreground membership 제약이 해당 board 목표를 위해 완화되어야 한다는 수학적 진술일 뿐, 새 foreground estimator만으로 충분하다거나 grouping이 양호하다는 인과 주장이 아니다. seed별 이질성 또는 generalization CI의 불확실성도 별도로 공개한다. support와 grouping을 동시에 바꾸는 실험을 자동 승인하지 않는다.

SW0165가 replication gate를 실패해도 위 진단은 세 completed seed 모두에 수행하되, 실패한 recipe를 확대하거나 promote하지 않는다. 어떤 결과도 SW0165의 frozen recipe, score gate, prediction 또는 protocol을 바꾸지 않으며 reserve/holdout 선택에 쓰지 않는다.
