# SW_0005 — background-cluster policy screen

Status: validation screen complete; no all-metric improvement.

## 쉽게 설명하면

SW_0001은 가장 큰 patch 그룹 하나만 배경으로 정했습니다. 그래서 실제
물체가 아닌 영역도 물체로 표시되는 경우가 많았습니다. 이번에는 **모델과
spike 결과는 그대로 두고**, 이미지 테두리를 많이 차지하는 그룹도 배경으로
보는 규칙을 시험했습니다. 미리 정한 20개 규칙을 같은 검증 이미지 320장에
적용했습니다. 이미지마다 정답을 보고 배경 그룹을 고르지는 않았습니다.

| spike 분류 후 배경 규칙 | 물체 구분 FG-ARI ↑ | 전경 IoU ↑ | 물체별 IoU ↑ |
|---|---:|---:|---:|
| 기존: 가장 큰 그룹만 배경 | 0.1363 | 0.2720 | 0.1456 |
| 전경 IoU 최고 규칙: 테두리 4 patch 이상, 그룹 내 테두리 비율 0.1 이상도 배경 | 0.0495 | 0.3220 | 0.1386 |

**해석:** 전경 영역은 더 잘 찾았지만, 서로 다른 물체를 구분하는 점수와
물체별 IoU는 낮아졌습니다. 세 지표를 함께 개선하지 못했으므로 이 규칙은
채택하지 않습니다. 20개 규칙의 전체 결과는 `results.json`에 있습니다.

The fixed-k=8 spike-synchrony classifier selects only its largest cluster as
background, leaving 45.8% of patches as predicted foreground on validation,
while the true foreground fraction is much lower. This experiment checks a
prediction-only perimeter rule: keep the original largest background cluster
and also designate a cluster as background if at least a fixed number of its
patches touch the image border and its own border fraction exceeds a fixed
threshold. `sweep.py` screens a small predeclared grid on the held-out
validation IDs 1320–1639 using saved SW_0001 predictions. It does not re-run
or modify S2Net, and no test-split predictions are used for selection.

Ground truth is read only to compute validation metrics after each fixed
prediction rule. The chosen rule, if any, must be locked before reference test
evaluation. A favorable score is a classifier change, not proof that theta or
spike representation improved.
