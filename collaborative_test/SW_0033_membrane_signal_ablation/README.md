# SW_0033: is the spatial readout using learned membrane information?

Status: 64-image pilot and full 320-image validation diagnostic complete.

## 핵심 결과

같은 320장과 같은 spectral k10에서 공간 kernel **단독** FG-ARI는
0.491272, 실제 membrane×공간은 0.494639였습니다. 따라서 SW_0028의
높은 FG-ARI를 현재 membrane 학습만의 성능으로 해석하면 안 됩니다.
membrane의 위치를 무작위로 섞으면 0.481408로 낮아져 영상별 정보가
완전히 없는 것은 아니지만, 공간 prior 대비 추가 이득은 매우 작습니다.

| affinity / validation 320 | FG-ARI ↑ | foreground IoU ↑ | object IoU ↑ |
|---|---:|---:|---:|
| spatial only, no model signal | 0.491272 | 0.190837 | 0.178026 |
| actual membrane × spatial | 0.494639 | 0.191070 | 0.170568 |
| permuted membrane × spatial | 0.481408 | 0.192980 | 0.165653 |

For true foreground pairs within three patch-grid units, raw membrane
affinity distinguishes same versus different objects with AUC 0.6365;
spatial distance alone reaches 0.6979 and membrane×spatial 0.6968.
This pairwise diagnostic uses GT *only for analysis*, never to construct a
predicted mask. Pair distributions are distance-confounded, so the AUC is
not an object-classification metric. All 320-image values are in
`validation320.json`; first-64 confirmation is in `pilot64.json`.

## Next decision

Keep the spatial readout as an ARI-first candidate, but prioritize training
signals that make the membrane affinity itself more object-specific. Any
such loss must be compared with matched checkpoint, seed, split, and
readout; do not attribute geometry's improvement to learned binding.

## 진단 목적

SW_0028의 공간 가중치가 FG-ARI를 높였지만, 모델이 물체에 맞는 membrane
패턴을 학습했기 때문인지 공간 거리만으로 나온 결과인지 분리할 필요가
있습니다. spatial-only, 실제 membrane×spatial, 그리고 membrane의
patch 위치를 무작위로 섞은 null을 같은 분류기로 비교합니다.

또한 실제 정답을 **진단에서만** 사용해 가까운 foreground patch 쌍의
membrane 유사도가 같은 물체와 다른 물체를 구별하는 AUC를 측정합니다.
정답은 어떠한 예측 mask 생성에도 사용하지 않습니다.

## Fixed setup

- Frozen SW_0003 seed-0 core; actual membrane last 192/256 steps.
- Same sigma1.5 16x16 spatial kernel, spectral k10, largest background.
- Fixed first 64 validation IDs 1320-1383 and shared patch evaluator.
- Random permutation seed 0 applies to patch-indexed membrane affinity
  independently per image, preserving matrix value distribution.
- Diagnostic pairs restricted to two foreground patches within 3 patch-grid
  units; report same/different means and rank-based AUC.
- No training, core, loss, production classifier, or metric change.

## Exact code change

Only this standalone diagnostic and launcher were added. The launcher now
accepts a validation count for exact 64- and 320-image reproduction.
