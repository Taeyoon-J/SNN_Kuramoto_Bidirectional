# SW176 결과: local phase credit 정책 종료

동일한 native32 고정320, seed1에서 학습·예측·채점을 완료하고 확정 예측으로 세 지표와 paired bootstrap을 독립 재계산했다. TRAIN512 두 번 노출의 supervised pilot이며 세 seed 평균 또는 70,000장 새 학습 결과가 아니다.

| 모델/출력 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| SW0164 | 0.770932 | 0.614993 | 0.630523 |
| SW0175_critical_component_conditional | 0.725044 | 0.614993 | 0.579214 |
| local_phase_conditional | 0.716674 | 0.614993 | 0.597555 |
| SW0175_critical_component_standalone | 0.674915 | 0.399856 | 0.509791 |
| local_phase_standalone | 0.709090 | 0.424322 | 0.590484 |

독립 출력은 SW175 대조군보다 FG-ARI가 0.03417 높아졌고 paired95% CI는 [0.01774, 0.05054]다. 그러나 기존 SW164 seed1보다 0.06184 낮으며 CI [-0.07551, -0.04838]다. 조건부 출력도 기존 모델보다 0.05426 낮고 CI [-0.06474, -0.04398]다. 따라서 등록한 두 endpoint의 확대 조건은 실패했다.

seed0/2 또는 70,000장으로 확대하지 않고 이 정책을 종료한다. 기존 최고 SW165는 유지한다. 단순히 finite gradient가 흐른다는 것만으로 분할 성능이 개선되지는 않았다. 이 결과는 모든 gating이 불필요하다는 증거는 아니다. 작은 LR/threshold/time-window 조정으로 같은 정책을 구제하지 않는다.

v1 채점의 standalone foreground 검사 오류는 GT를 열기 전에 발견했다. 대기 중인 CPU scorer만 중지하고, 원본 학습·예측·코드 해시를 보존한 채 v2 scorer를 배포했다. 로컬·서버 각각 13개 검사 후 같은 확정 예측을 채점했다. 자세한 이력은 EVALUATION_QUEUE_KO.md에 있으며 metric·bootstrap·확대 조건은 바꾸지 않았다.

다음 후보는 이미지별로 학습되는 edge polarity와 동일 크기의 nonnegative 대조군 비교다. 기존의 균일 반발 개입 실패나 동료의 pending 결과를 성공 근거로 취급하지 않는다. 실제 negative interaction 사용과 최종 mask 개선을 함께 확인해야 한다. 아직 이 후보의 실제 GPU 학습 결과는 없다.

검증된 score SHA: `7a4c4a3fcc0b00dd210703133bea6f687aef72b82fc3937cf5d4603eaaba4ab6`. 원본 score와 독립 검증·최종 모델 lineage는 `results_archive/completed_score_v2_20261016/`에 보존했다.
