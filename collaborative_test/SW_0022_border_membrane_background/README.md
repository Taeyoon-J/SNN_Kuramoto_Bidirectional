# SW_0022: membrane-border prototype for background selection

Status: fixed 320-image validation complete. Small all-metric improvement;
retain as a candidate, but foreground/background remains the bottleneck.

## 쉽게 설명하면

SW_0021은 물체 그룹 수와 FG-ARI를 개선했지만 배경 patch를 너무 많이
물체로 취급했습니다. 가장 큰 slot 전체를 배경으로 버리는 대신, 이미지
테두리 patch들의 membrane 시간 패턴을 기준으로 배경 patch를 고르는
방법을 시험합니다.

## Exact setup

- Keep the SW_0003 seed-0 40-epoch checkpoint and SW_0021's adaptive
  membrane slots fixed: 256 steps, discard 64, cosine threshold .7, six
  initial slots, deterministic assignment seed 0.
- Derive one normalized mean temporal pattern from the 60 border patches.
  Cosine similarity of every membrane trace to that border prototype is a
  candidate background score. This uses only model membrane output and patch
  positions, not ground-truth masks or RGB.
- Compare the existing largest-slot rule with template-only, union, and
  intersection rules at thresholds .2/.4/.6/.8/.9/.95/.98. Thresholds are
  selected on validation only. The fixed HDF5 validation IDs are 1320-1639,
  same 16x16 patch evaluation contract.
- Report all three metrics, predicted group count, foreground fraction,
  precision, and recall. GT may label the diagnostic score distribution and
  metrics only; it never enters mask creation.

## Code changes

Only this standalone evaluation script and report; the core, training losses,
and production classifier are unchanged.

## Results and insight

The same SW_0003 checkpoint and 320 validation images were used throughout.

| Background rule | FG-ARI | Foreground IoU | Matched-object IoU | Predicted FG fraction |
|---|---:|---:|---:|---:|
| Largest slot only (SW_0021) | .357053 | .209870 | .113992 | .6583 |
| Largest slot **and** border-template similarity >= .90 | **.361770** | **.215081** | **.119675** | .6771 |
| Largest slot and similarity >= .95 | .356953 | .223897 | .126147 | .7189 |

The `.90` intersection rule raises all three metrics a little, so it remains
an improvement candidate. The `.95` rule yields better IoUs while essentially
holding the earlier FG-ARI, so it is another useful tradeoff candidate.
Neither is final: GT foreground occupies only .2168 of patches, whereas the
`.90` prediction labels .6771 as foreground. On validation GT-labeled pairs,
mean border-template cosine was .8957 for true background and .8584 for true
foreground. This weak score separation explains why a simple membrane-border
template cannot remove enough background safely. GT labels were used **only**
for this post-hoc diagnostic and metrics; the prediction uses membrane alone.
Full 22-row sweep is in `validation320.json`.

Next: inspect whether other *actual membrane/spike features* distinguish
foreground/background more sharply, while preserving SW_0021's strong
within-foreground grouping. Do not treat the tiny gain as proof of goal
attainment; it is seed-0 validation only.
