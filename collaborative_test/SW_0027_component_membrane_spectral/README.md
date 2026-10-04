# SW_0027: per-component membrane affinity for spectral grouping

Status: 64-image pilot and full 320-image seed-0 validation complete;
classifier-only comparison, no production merge.

## 결과

FG-ARI를 가장 높게 유지한 것은 기존의 component 평균 membrane 유사도입니다.
component별 양수 상관계수의 평균은 FG-ARI가 약간 낮지만 두 IoU가 함께
올랐습니다. 하나의 지표 감소만으로 버리지 않고 IoU 후보로 보존합니다.

| readout, full validation 320 | FG-ARI ↑ | foreground IoU ↑ | matched-object IoU ↑ | groups/image |
|---|---:|---:|---:|---:|
| 기존 aggregate absolute (FG-ARI lead) | 0.393974 | 0.228821 | 0.150706 | 9.0 |
| component positive mean (IoU track) | 0.386094 | 0.232912 | 0.157926 | 9.0 |
| component absolute mean | 0.368658 | 0.230698 | 0.155737 | 9.0 |
| component absolute product | 0.368075 | 0.231474 | 0.142292 | 9.0 |
| component positive product | 0.371997 | 0.229641 | 0.140427 | 9.0 |

The 64-image pilot also showed the same qualitative tradeoff: aggregate
absolute had top FG-ARI .359342, while component positive mean raised both
IoUs (.213954 to .224100 foreground and .139526 to .151059 object). Full
tables are in `pilot64.json` and `validation320.json`. These are not
three-seed estimates, nor a reference-test or Slot Attention win.

## 실험 목적

현재 FG-ARI 선두인 SW_0024는 네 membrane component를 평균한 뒤
oscillator 간 시간 패턴의 유사도를 계산합니다. 이 평균 과정에서 component별
구분 정보가 사라질 수 있습니다. 동일한 checkpoint와 동일한 검증 이미지를
유지한 채 component별 유사도를 각각 계산하고 결합하는 방법을 비교합니다.
한 지표가 떨어져도 FG-ARI 또는 IoU 개선 후보는 별도로 보존합니다.

## Contract

- Frozen SW_0003 seed-0 checkpoint; no retraining or core/loss changes.
- Actual membrane histories, last 192 of 256 steps, 16x16 oscillators.
- Fixed validation IDs 1320-1639; first 64 pilot, then full 320 only if useful.
- Same normalized spectral clustering, fixed k=10, largest cluster as
  background, same patch-level GT/evaluator as SW_0024.
- Control: absolute centered correlation of component-averaged membrane.
  Candidates: mean/product of absolute or positive component correlations.
- No GT or oracle count used for mask generation. No reference-test tuning.

## Code changes

Only this folder's standalone evaluator and launcher were added. The spectral
normalization, deterministic spherical k-means, largest-cluster background,
checkpoint, core, training code, loss, and fixed patch evaluator are unchanged.
