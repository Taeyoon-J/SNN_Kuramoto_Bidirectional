# SW_0031: independent membrane foreground mask plus spatial groups

Status: 64-image pilot and full 320-image seed-0 validation complete;
not adopted as FG-ARI lead.

## 결과와 판단

동일한 membrane history에서 얻은 비공간 foreground mask를 공간 그룹에
patch별로 곱하면 foreground IoU는 조금 높아지지만 FG-ARI와 object IoU가
모두 떨어집니다. 전역적으로 거르는 방식은 현재 선두를 대체하지 않고,
foreground-IoU 후보로만 남깁니다. 배경 가능성이 높은 patch에만
제한하여 거르는 후속 실험이 필요합니다.

| readout / full validation 320 | FG-ARI ↑ | foreground IoU ↑ | object IoU ↑ | predicted FG fraction |
|---|---:|---:|---:|---:|
| spatial control | 0.494639 | 0.191070 | 0.170568 | 0.8676 |
| nonspatial positive control | 0.386094 | 0.232912 | 0.157926 | 0.7312 |
| spatial AND nonspatial aggregate | 0.446381 | 0.203298 | 0.161697 | 0.6774 |
| spatial AND nonspatial positive | 0.432699 | 0.205366 | 0.161077 | 0.6585 |

The true foreground fraction is 0.2168. First-64 pilot showed the same
tradeoff; all rows are in `pilot64.json` and `validation320.json`. No
three-seed/reference-test or Slot Attention superiority claim follows.

## 실험 목적

SW_0028의 공간 가중치 그룹은 FG-ARI가 높지만 예측 foreground가 너무
넓습니다. 동일한 membrane history에서 계산한 비공간 spectral 분류기는
foreground IoU가 더 높습니다. 비공간 분류기의 foreground 여부만 사용해
공간 그룹을 patch별로 gate하면 두 장점을 합칠 수 있는지 확인합니다.

## Exact setup

- Frozen SW_0003 seed-0 core and actual membrane, last 192/256 steps.
- Fixed CLEVR IDs 1320-1639, first 64 pilot, patch grid16.
- Group IDs: SW_0028 spatial Gaussian sigma1.5, spectral k10.
- Candidate foreground masks: no-spatial aggregate absolute affinity or
  no-spatial component positive-mean affinity, both spectral k10.
- Hybrid prediction: keep spatial group ID where the independent membrane
  classifier says foreground, otherwise assign background ID 0.
- Score both source controls and both hybrids on identical images. No
  ground truth or reference-test data enters any prediction.
- No training, core, loss, or production classifier change.

## Exact code change

Only this standalone evaluator and launcher were added. No model,
training, loss, production classifier, or patch evaluator code changed.
