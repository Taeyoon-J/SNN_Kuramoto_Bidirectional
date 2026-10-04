# SW_0003 — matched-duration membrane PLV

Status: 40-epoch training and fixed-split validation complete. No all-metric improvement.

## 쉽게 설명하면

SW_0002에서 membrane을 기준으로 학습했을 때 물체 구분 점수는 좋아졌지만,
기존 모델은 40 epoch, 새 모델은 10 epoch만 학습해 공정한 비교가
아니었습니다. 이번에는 **membrane 기준으로 똑같이 40 epoch** 학습했습니다.
나머지 설정은 기존 모델과 같습니다. 따라서 이번에는 학습 길이가 같은 두
모델을 같은 검증 이미지 320장으로 비교했습니다.

| 같은 40 epoch, spike 패턴 분류 | 물체 구분 FG-ARI ↑ | 전경 IoU ↑ | 물체별 IoU ↑ | 실제 spike 비율 |
|---|---:|---:|---:|---:|
| 기존 위상 기준 학습 | 0.1363 | 0.2720 | 0.1456 | 0.4849 |
| 새 membrane 기준 학습 | 0.2889 | 0.2075 | 0.1027 | 0.4159 |

**해석:** membrane을 기준으로 학습하자 서로 다른 물체를 구분하는 점수는
0.1526 올랐지만, 물체가 실제 어디 있는지를 맞추는 두 점수는 더
떨어졌습니다. 학습 길이를 같게 해도 같은 방향의 손익이 나타났으므로
이 설정은 목표인 **세 점수 모두 개선**에 실패했습니다. 기존의 평균 활동량
분류법은 두 모델 모두 화면 전체를 물체 하나로 잡아 세 점수가 거의
동일했습니다(0.0000 / 0.2168 / 0.0148).

The SW_0002 pilot changed PLV source and used only 10 epochs, while the starting
checkpoint used 40. This follow-up repeats SW_0002 at seed 0 for 40 epochs to
isolate the source change at matched training duration. All other settings are
copied verbatim from the seed-0 baseline. See `run.sh`.

Compare the original spatial-components classifier and the SW_0001 fixed-k=8
spike-synchrony classifier on validation IDs 1320–1639, with the same patch
metrics. No test split is used for selection. Training loss alone is not a
success criterion.

Full per-image validation scores are in `results.json`; raw patch predictions
remain in the server output directory `validation_fixed_split/patch_masks.pt`.
