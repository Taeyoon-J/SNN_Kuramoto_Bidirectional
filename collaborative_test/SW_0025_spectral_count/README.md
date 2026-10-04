# SW_0025: spectral cluster-count follow-up

Status: 64-image pilot and full 320-image validation complete. k=10 remains
the FG-ARI lead; larger k improves object IoU and remains a secondary track.

## 쉽게 설명하면

SW_0024의 spectral clustering은 시험한 최대 그룹 수 k=10에서 가장
높은 FG-ARI를 냈습니다. 더 많은 그룹을 허용하면 좋아지는지, 아니면
물체를 지나치게 쪼개 점수가 떨어지는지 확인합니다.

## Controlled settings

- Same frozen SW_0003 seed-0 core, actual membrane histories, 256-step
  rollout minus first 64, and the SW_0024 spectral implementation.
- Only k changes: 10/12/14/16/20. Both absolute and positive-only membrane
  correlation are recorded. k=10 is the exact SW_0024 control.
- First 64 validation IDs 1320-1383 for pilot, then all 320 only if a useful
  setting emerges. No GT object count is used to set k per image. GT masks
  are only for patch-level scoring. No reference-test use during selection.
- Primary metric FG-ARI; foreground and object IoUs and predicted group counts
  must be reported alongside.

## Code changes

`SW_0024_membrane_spectral/evaluate.py` receives optional
`--cluster-counts`; its default [4,6,8,10] exactly preserves SW_0024.
This experiment adds only its run command and report. No model/loss change.

## Results and decision

On the first 64 validation images, absolute-correlation k=12 slightly raised
FG-ARI .359342 to .362888 versus k=10, and object IoU .139526 to .161189.
This pilot effect did **not** hold in the full 320-image validation:

| Absolute membrane spectral | FG-ARI | FG IoU | Object IoU | Groups/image |
|---|---:|---:|---:|---:|
| k=10 (SW_0024 control) | **.393974** | .228821 | .150706 | 9 |
| k=12 | .381903 | .227836 | .169208 | 11 |
| k=14 | .382332 | .230189 | .194429 | 13 |
| k=16 | .368964 | .232346 | .209930 | 15 |
| k=20 | .348207 | .232265 | **.228019** | 19 |

The k=10 setting remains the ARI-first candidate. k=20 nearly reaches the
saved Slot reference object-IoU number .235487, but that reference is on a
different held-out split, so this is **not** a direct success comparison.
k=20 badly overfragments objects (19 predicted groups/image versus true mean
6.20) and reduces FG-ARI. Preserve it as an object-IoU-focused readout
component, not as the new overall default. Positive-only correlations show
the same tradeoff. Full pilot and validation sweeps are saved separately;
there is no three-seed result and no production classifier change.
