# SW_0028: spatially weighted membrane spectral grouping

Status: 64-image pilot and full 320-image seed-0 validation complete;
classifier-only result, no production merge.

## 결과와 해석

Gaussian 공간 가중치 sigma=1.5 patch가 FG-ARI를 크게 올렸습니다.
같은 checkpoint와 같은 320장 검증에서 기존 0.393974에서 0.494639로
상승했습니다. matched-object IoU도 상승하지만 foreground IoU는 하락합니다.
따라서 FG-ARI 선두 후보로 유지하고, foreground/background 분리는 별도
개선 대상으로 남깁니다. 한 지표 하락만으로 이 후보를 기각하지 않습니다.

| readout / 320 images | FG-ARI ↑ | foreground IoU ↑ | matched-object IoU ↑ | predicted groups/image |
|---|---:|---:|---:|---:|
| control, no spatial weighting | 0.393974 | 0.228821 | 0.150706 | 9.0 |
| sigma 1.5, new FG-ARI lead | 0.494639 | 0.191070 | 0.170568 | 9.0 |
| sigma 3 | 0.475584 | 0.197766 | 0.167015 | 9.0 |
| sigma 6 | 0.462739 | 0.191333 | 0.161674 | 9.0 |
| sigma 12 | 0.427110 | 0.197835 | 0.158321 | 9.0 |
| sigma 24 | 0.404347 | 0.222726 | 0.162998 | 9.0 |

First-64 pilot also showed sigma 1.5 best FG-ARI (.530110 versus .359342
control) and lower foreground IoU (.184325 versus .213954). The full
320-image result confirms the direction, albeit smaller magnitude.
The number of output groups remains fixed by k=10 and largest-cluster
background; 9 groups per image. This is seed-0 validation only, not a
three-seed/reference-test goal result. Full scores are in `pilot64.json`
and `validation320.json`.

## 실험 목적

SW_0024의 membrane 시간 패턴은 서로 멀리 있는 두 patch도 유사하면 같은
그룹으로 묶을 수 있습니다. 이 경우 물체 사이의 잘못된 결합이 FG-ARI를
낮출 수 있습니다. 동일한 spike/membrane 학습 checkpoint, 이미지, k=10
분류기를 유지하고, 유사도에 2D patch 거리의 Gaussian weight만 곱합니다.
FG-ARI를 먼저 보되, IoU만 개선한 결과도 별도 후보로 유지합니다.

## Exact contract

- Frozen SW_0003 seed-0 core; 256 steps, first 64 discarded.
- Actual component-averaged membrane; absolute centered temporal correlation.
- 16x16 grid; sigma in patch units: 1.5, 3, 6, 12, 24; no-spatial control.
- Same normalized spectral clustering, k=10, largest cluster background.
- Fixed CLEVR HDF5 validation IDs 1320-1639, first 64 pilot then full 320
  only if useful. Same patch-level GT and evaluator as SW_0024.
- No GT, oracle object count, or reference-test information enters prediction.
  No retraining or core/loss change.

## Exact code change

Only this folder's standalone readout evaluator and launcher were added.
The new readout multiplies the existing membrane affinity matrix by
`exp(-||patch_i-patch_j||^2 / (2*sigma^2))` before the existing normalized
spectral clustering. No core, loss, optimizer, training, or patch metric code
changed. The no-spatial control reproduces SW_0024's full-validation scores.
