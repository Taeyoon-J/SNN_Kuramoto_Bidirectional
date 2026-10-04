# SW_0001 — spike synchrony readout

Status: completed validation probe; no model weights or core computation changed.

## 쉽게 설명하면

기존 방법은 각 patch의 활동량을 시간에 걸쳐 평균낸 뒤 물체를 찾았는데,
320장의 검증 이미지에서 거의 화면 전체를 물체 하나로 묶었습니다. 이번에는
**학습된 모델은 그대로 두고**, 두 patch가 256개 시간 단계 동안 비슷한 시점에
spike를 내는지 비교했습니다. 비슷한 patch를 8개 그룹으로 묶고 가장 큰 그룹을
배경으로 정했습니다. 정답의 물체 개수는 분류에 넣지 않았습니다.

| 같은 검증 이미지 320장 | 물체 구분 FG-ARI ↑ | 전경 IoU ↑ | 물체별 IoU ↑ |
|---|---:|---:|---:|
| 기존 평균 활동량 방식 | 0.0000 | 0.2168 | 0.0148 |
| 새 spike 패턴 방식 | 0.1363 | 0.2720 | 0.1456 |

**해석:** 세 지표가 모두 좋아졌습니다. 다만 여전히 Slot Attention 기준에는
못 미치고, 이는 모델을 새로 학습한 결과가 아니라 *결과를 읽는 방법*만 바꾼
seed 0 검증 결과입니다.

## Hypothesis

The starting spatial-components classifier returns one object for every evaluated image because time-mean sigmoid membrane activity is above 0.5 almost everywhere. Spike rhythm may still contain pairwise object information. A fixed-count clustering of centered spike traces could recover more than one object.

## Procedure

Use the existing seed-0 baseline checkpoint. Run 256 time steps, discard the first 64 for similarity, and compute the absolute centered cosine similarity between oscillators' returned spike traces. Apply normalized spectral clustering with a fixed k=8 and designate the largest cluster as background. Other clusters become object IDs. The true object count is never supplied to the classifier. Evaluate on validation IDs 1320–1639.

Candidate sources and k=4, 6, 8 were screened on the first 16 validation images. The aggregate spike affinity with k=8 was then checked on the full 320-image validation split. A component-wise product affinity with k=6 was retained as a contrast. The full-validation scores are below.

| Readout | Patch FG-ARI | Foreground IoU | Matched object IoU | Predicted foreground fraction |
|---|---:|---:|---:|---:|
| Aggregate spike synchrony, k=8 | 0.136286 | 0.271997 | 0.145604 | 0.458057 |
| Product of four component synchronies, k=6 | 0.055590 | 0.282826 | 0.121303 | 0.169727 |

The same-split baseline spatial-components scores are 0.000000 / 0.216821 /
0.014756 on validation IDs 1320–1639. Thus the fixed-k spike-synchrony readout
improves all three validation metrics against the original classifier with
identical model weights and examples. The separate reference test result
(0.000000 / 0.207996 / 0.015036) is not used for this pairwise comparison.
The reproducible evaluation is `../evaluate_fixed_split.py`; full same-split
results are stored under the baseline checkpoint's `validation_fixed_split`.

## Diagnostic

On the first 16 validation images, mean phase PLV was 0.444 within a true object and 0.388 across true objects. Spike synchrony was 0.462 within an object and 0.389 across objects; background-background synchrony was 0.679. This indicates object information survives into spike timing, although separation is weak. The original classifier returned exactly one object per image in this diagnostic set.

Next test: measure SNN gradient connectivity under the phase-only objective, then introduce a narrowly scoped spike-path training signal. Keep this readout fixed while testing the loss.
