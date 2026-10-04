# SW_0029: refine spatial scale and spectral group count

Status: 64-image validation pilot complete; no full-320 promotion because no
candidate beat the existing FG-ARI lead and the small object-IoU differences
are too weak to justify a new readout choice.

## 결과와 판단

sigma=1.5, k=10이 64장에서도 FG-ARI 최고입니다. k=6이나 k=8로
그룹 수를 줄여도 물체 분리는 개선되지 않았습니다. sigma=1.25, k=10은
matched-object IoU가 0.176630에서 0.178952로 소폭 올랐지만,
FG-ARI는 0.530110에서 0.520789로 떨어졌습니다. 이 작은 IoU 차이는
예비 64장 자료로만 기록하고 채택하지 않습니다. SW_0028의 320장
sigma=1.5, k=10 선두 결과를 유지합니다.

| setting / first 64 images | FG-ARI ↑ | foreground IoU ↑ | matched-object IoU ↑ |
|---|---:|---:|---:|
| sigma 1.5, k10 control | 0.530110 | 0.184325 | 0.176630 |
| sigma 1.25, k10 | 0.520789 | 0.184780 | 0.178952 |
| sigma 1, k10 | 0.513116 | 0.182321 | 0.172746 |
| sigma 1.5, k8 | 0.462543 | 0.185843 | 0.131978 |
| sigma 2, k8, best foreground IoU | 0.453026 | 0.195618 | 0.135411 |

All 15 rows are in `pilot64.json`. This is seed-0 validation only, not a
three-seed test or evidence that the final goal has been met.

## 실험 목적

SW_0028에서 sigma=1.5 patch의 공간 가중치가 FG-ARI를 크게 올렸습니다.
이번에는 더 짧은 거리 범위와 그룹 수 6/8/10을 같은 학습 모델에 적용해
object 분리가 더 좋아지는지 확인합니다. FG-ARI가 우선이지만 다른 지표의
개선도 별도 후보로 보존합니다.

## Fixed evaluation contract

- Frozen SW_0003 seed-0 core, actual membrane history, last 192/256 steps.
- Fixed 16x16 CLEVR patch labels and IDs 1320-1639.
- Same normalized spectral eigendecomposition, deterministic spherical k-means,
  and largest-cluster background rule as SW_0028.
- Pilot: first 64 validation images, sigma .75/1/1.25/1.5/2 patch units,
  cluster counts 6/8/10. The sigma 1.5, k10 point is the SW_0028 control.
- No GT-derived object count, no retraining/core/loss change, and no
  reference-test threshold selection.

## Exact code change

Only this standalone sweep and launcher were added. No model, loss,
training, production classifier, or evaluator code changed.
