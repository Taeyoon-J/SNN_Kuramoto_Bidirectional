# SW_0021: ARI-first follow-up on the membrane-PLV core

Status: full 320-image classifier-only validation complete. FG-ARI improved;
retain this branch of experiments despite low IoUs.

## 쉽게 설명하면

SW_0003은 물체 구분 점수(FG-ARI)가 높았지만 다른 두 점수는 낮았습니다.
그 결과를 버리지 않고, 같은 학습된 코어에 사용자 제안 adaptive slot
분류를 적용해 물체 구분을 더 높일 수 있는지 확인합니다.

## Controlled setup

- Frozen SW_0003 seed-0 40-epoch membrane-PLV checkpoint; no retraining or
  core/loss modification.
- Fixed CLEVR HDF5 validation IDs 1320-1639, 16x16 patch evaluation; no test
  IDs and no GT masks in predictions.
- Existing SW_0013 adaptive cosine slot classifier, evaluated separately on
  actual spike histories and continuous membrane histories, 256-step rollout
  with first 64 discarded.
- Thresholds .3/.5/.7/.9, initial slot counts 3/6, deterministic assignment
  seed 0. The largest slot is background, as in SW_0013. No oracle object
  counts or per-image threshold selection.
- Primary comparison is FG-ARI versus SW_0003's fixed-k spike-spectral
  validation score .288924, while reporting both IoUs and group counts.
  If ARI rises, retain the method for later foreground/IoU work even if one
  IoU falls; final success still requires three-seed all-metric superiority.

## Code changes

Only this run script and report; the existing evaluator/classifier and trained
core are reused unchanged.

## Results and next step

| Readout on the same SW_0003 checkpoint | Patch FG-ARI | Foreground IoU | Matched-object IoU | Groups/image |
|---|---:|---:|---:|---:|
| Prior fixed-k spike spectral | .288924 | .207549 | .102696 | fixed k=8 |
| Adaptive slots, actual spikes, threshold .9/six initial slots | .313070 | .202942 | .107462 | 6.37 |
| Adaptive slots, membrane, threshold .7/six initial slots | **.357053** | .209870 | .113992 | 5.98 |

All rows use validation IDs 1320-1639, seed-0 checkpoint, and the same
patch-level evaluator. Mean true foreground object count is 6.20/image.
The membrane-slot rule raises FG-ARI by .068129 over the prior readout and
predicts a near-correct *average* group count. Exact group count is still
only 11.9% of images, so average agreement is not count accuracy. Foreground
IoU remains around .21, below the saved Slot reference .212251 on a separate
reference split, and matched-object IoU remains far below .235487. Those
split-specific numbers are not a direct performance comparison.

The high FG-ARI is a useful intermediate result, not a goal claim. Keep this
checkpoint/readout as the **ARI-first track**. Its predicted foreground
fraction is .6583, suggesting background selection may be a bottleneck;
the next test should diagnose and improve foreground selection without
discarding the strong membrane grouping. Full sweeps are in
`spikes_validation320.json` and `membrane_validation320.json`.
