# SW_0030: border-aware background after spatial membrane grouping

Status: first-64 validation pilot complete; no full-320 promotion.

## 결과와 판단

기존 sigma1.5/k10 결과는 예측 foreground가 patch의 86.5%인데 실제는
19.9%입니다. 테두리 그룹을 추가 배경으로 처리해 예측 foreground를
22.8%로 맞춰도 위치가 맞지 않아 foreground IoU는 0.184→0.198로
조금만 오르고 FG-ARI는 0.530→0.261로 크게 떨어집니다. 따라서
테두리 접촉만으로 그룹 전체를 배경 처리하는 규칙은 채택하지 않습니다.

| setting / first 64 validation | FG-ARI ↑ | foreground IoU ↑ | object IoU ↑ | predicted FG fraction |
|---|---:|---:|---:|---:|
| original spatial grouping | 0.530110 | 0.184325 | 0.176630 | 0.8648 |
| border fraction >=0.2 | 0.261012 | 0.198330 | 0.080716 | 0.2283 |
| border fraction >=0.4 | 0.514775 | 0.188543 | 0.166820 | 0.7776 |

All seven rows are in `pilot64.json`; no metric claim beyond this pilot.
The experiment isolates foreground selection: reaching the correct *amount*
of foreground does not imply finding the correct foreground patches. The
underlying spatial membrane grouping still leads FG-ARI, so keep it.

## 실험 목적

현재 FG-ARI 선두인 SW_0028은 하나의 가장 큰 spectral cluster만 배경으로
처리합니다. 배경이 여러 그룹으로 나뉘었다면 나머지가 잘못 foreground로
남아 foreground IoU를 낮춥니다. 학습 모델과 그룹 구조는 그대로 두고
테두리와 닿은 그룹을 배경으로 추가하는 규칙을 비교합니다.

## Fixed setup

- Frozen SW_0003 seed-0 core; actual membrane history, last 192/256 steps.
- SW_0028 spatial Gaussian sigma1.5, normalized spectral k10, largest
  cluster background, 16x16 patch grid. No core or affinity changes.
- Additional whole-group background rule: fraction of group patches at
  image border >= threshold, with at least one border patch.
- Thresholds 0, .05, .1, .2, .4, .6; fixed validation IDs 1320-1639,
  first 64 pilot and full 320 only if useful.
- No GT/background labels, oracle object count, or reference-test information
  enter prediction. Same three patch metrics as every SW experiment.

## Exact code change

Only this standalone evaluator and launcher were added. No model,
affinity, training, loss, production classifier, or patch metric changed.
