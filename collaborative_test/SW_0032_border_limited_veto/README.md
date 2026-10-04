# SW_0032: border-limited membrane foreground veto

Status: first-64 pilot and full 320-image seed-0 validation complete;
an IoU-improving tradeoff, not a new FG-ARI lead.

## 결과와 판단

전체를 거르던 SW_0031과 달리 이미지 가장자리에서만 독립적인 membrane
foreground 판정을 적용하면 FG-ARI 손실을 줄이면서 두 IoU를 개선합니다.
가장자리 두 patch 폭에서 FG-ARI는 0.495→0.468로 조금 내려가지만
foreground IoU는 0.191→0.210, object IoU는 0.171→0.174로 올라갑니다.
이 후보는 IoU track에 보존하고, sigma1.5/k10 원본을 FG-ARI lead로
유지합니다. 이 작은 상승을 최종 성능으로 주장하지 않습니다.

| readout / full validation 320 | FG-ARI ↑ | foreground IoU ↑ | object IoU ↑ | predicted FG fraction |
|---|---:|---:|---:|---:|
| spatial control | 0.494639 | 0.191070 | 0.170568 | 0.8676 |
| border width 1 | 0.480201 | 0.202005 | 0.172553 | 0.8032 |
| border width 2, two-IoU tradeoff | 0.467718 | 0.210417 | 0.174335 | 0.7477 |
| border width 3, highest foreground IoU | 0.452357 | 0.213411 | 0.172639 | 0.7024 |
| width 8, blanket SW_0031 intersection | 0.432699 | 0.205366 | 0.161077 | 0.6323 |

The true foreground fraction is 0.2168, so false foreground remains large.
The pilot64 and validation320 JSONs retain every tested width. All scores
are seed-0 validation, not three-seed/reference-test goal evidence.

## 실험 목적

SW_0031의 전역적인 foreground veto는 IoU를 높이지만 FG-ARI를 크게
낮췄습니다. 실제 물체가 주로 이미지 내부에 있다는 약한 위치 prior를
사용해, 테두리에서만 그 veto를 적용하면 물체 분리를 덜 손상할 수 있는지
확인합니다. GT label이나 true foreground fraction은 예측에 쓰지 않습니다.

## Fixed setup

- Same frozen SW_0003 seed-0 core and actual membrane history.
- Spatial group IDs: SW_0028 Gaussian sigma1.5, spectral k10.
- Independent foreground veto: SW_0027 component-positive mean, k10.
- Veto only at patches with minimum grid distance to image border smaller
  than 1, 2, 3, 4, 6, or 8; width8 equals SW_0031 blanket intersection.
- Same fixed CLEVR validation IDs and 16x16 patch evaluator. First 64 pilot,
  full320 only if a useful candidate emerges.
- No core, loss, optimizer, training, production classifier, or metric change.

## Exact code change

Only this folder's standalone evaluator and launcher were added. The veto
uses a deterministic grid-distance condition plus an independently derived
membrane mask; it does not change the saved core or any production module.
