# SW_0024: membrane spectral grouping pilot

Status: 64-image pilot and full fixed 320-image validation complete. New
ARI-first candidate; not yet a three-seed or reference-test result.

## 쉽게 설명하면

정답으로 배경을 완벽하게 분리해도 FG-ARI가 오르지 않았습니다. 따라서
물체 patch들을 묶는 방식 자체를 바꿔 봅니다. 동일한 membrane history와
동일한 이미지에서 기존 adaptive slot과 spectral clustering을 비교합니다.

## Exact setup

- Freeze SW_0003 seed-0 40-epoch core. Use only actual membrane histories,
  256-step rollout and first 64 discarded. First 64 fixed validation IDs
  1320-1383. No reference-test images or GT in mask creation.
- For each patch, center and normalize its temporal membrane trace. Compare
  absolute correlation and positive-only signed correlation as the affinity.
- Normalized adjacency eigendecomposition and the repository's deterministic
  spherical k-means, testing fixed k=4/6/8/10. Largest spectral group is
  background. k is a validation-selected hyperparameter, never the GT count.
- Evaluate the SW_0021 adaptive-slot rule on exactly the same 64 images as a
  control. Primary target is FG-ARI; record both IoUs and group counts.
  Promising settings must be confirmed on all 320 validation images before
  further claims.

## Code changes

Only this standalone evaluator and report. No core or training change.

## Results

The 64-image pilot selected absolute membrane correlation with k=10:
FG-ARI .359342 / foreground IoU .213954 / object IoU .139526, versus
adaptive slots .316233 / .195784 / .109259 on exactly those images.

On all 320 fixed validation images, the identical k=10 rule scored:

| Readout on the same SW_0003 core | FG-ARI | Foreground IoU | Object IoU | Predicted groups/image |
|---|---:|---:|---:|---:|
| Adaptive membrane slots (SW_0021) | .357053 | .209870 | .113992 | 5.98 |
| Absolute-correlation spectral, k=10 | **.393974** | **.228821** | **.150706** | 9.00 |
| Positive-only spectral, k=10 | .391096 | .230472 | .151155 | 9.00 |

All three scores improved for the validation-selected absolute-correlation
readout. This is a genuine classifier-only gain on one seed, not proof of the
goal: the saved Slot reference is on a different reference split and the
three-seed contract remains unmet. The fixed-k classifier creates exactly
nine predicted foreground groups/image after removing its largest cluster,
above the true mean 6.20, so there is likely remaining fragmentation.

The grouping algorithm is now a stronger ARI-first candidate than adaptive
slots on this checkpoint. Both 64- and 320-image results are preserved in
`pilot64.json` and `validation320.json`. Next test can examine whether a
different non-oracle cluster count or graph/affinity improves FG-ARI further,
while separately addressing foreground precision.
