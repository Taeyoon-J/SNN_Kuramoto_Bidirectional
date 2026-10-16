# SW177 최종 결과: 기존 모델 개선 실패

고정 native32 평가320장, seed1의 supervised TRAIN512×2 passes/64 updates 비교다. 기존 최고 세 seed 평균이나 새70,000장 학습 결과가 아니다. 두 모델의 학습 checkpoint·Adam·순서와 불변모델 예측 receipt를 검증하고, 양쪽320 예측을 확정한 뒤 GT를 열었다. Production 재평가와 별도 metric/평균/bootstrap/gate 재계산이 일치했다.

| 모델 / 출력 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 SW164 seed1 | 0.770932 | 0.614993 | 0.630523 |
| Signed / 원본 foreground 유지 | 0.725166 | 0.614993 | 0.613633 |
| Nonnegative / 원본 foreground 유지 | 0.693004 | 0.614993 | 0.602618 |
| Signed / 자체 native-QCC foreground | 0.718381 | 0.388679 | 0.600379 |
| Nonnegative / 자체 native-QCC foreground | 0.680424 | 0.471769 | 0.590524 |

Signed의 기존 모델 대비 FG 차이는 원본 foreground 유지 시 −0.045766, paired95% CI [−0.054653,−0.037457], 자체 foreground에서는 −0.052551, CI [−0.063636,−0.041754]다. Bootstrap은 같은320 이미지·10,000 resamples·seed177이다.

Signed는 새 nonnegative control보다 FG가 높았다(conditional +0.032162, standalone +0.037957, 두 CI 하한>0). 그러나 기존 모델보다 낮았고 자체 foreground IoU는 control보다도 낮았다. 등록된 종합 조건을 통과하지 못했으므로 최고 모델로 승격하지 않고 seed0/2·70k 확장도 하지 않는다.

## 학습 신호와 성능을 구분

실제 GPU 사전 검증은 원본과 zero-head signed의 forward/mask exact parity, 실제 emitted S→head gradient를 통과했다. 두 실제 첫 committed updates에서도 spike/head credit과 parameter 변경을 확인했다. 이러한 학습 작동 증거가 최종 mask 개선을 보장하지는 않았다.

TRAIN8 loss는 초기0.3869964841에서 signed0.3674587768, nonnegative0.3204177671로 내려갔다. Loss가 더 낮았던 control의 FG가 더 나빴다. 현재 절대 pair/forest 목적함수와 최종 mask 구분 사이의 불일치가 다음 검증 가설이며, 이번 결과만으로 정확한 원인을 모두 확정하지 않는다.

Signed의 최종 TRAIN8에서 negative edge fraction과 실제 negative-kernel contribution RMS는 모두0이었다. 이것은8장 관측으로, 전체 TRAIN/VAL의 음수 사용 부재를 증명하지 않는다. Signed/control 성능 차이를 repulsion 또는 모든 gating의 기여로 해석하지 않는다. 실패 후 bias/LR/온도/passes를 바꿔 이 recipe를 연장하지 않는다.

다음 후보는 실제 spike temporal code의 상대적인 object-class competition이다. 전체 TRAIN modal patches와 BG를 loss에서 사용하고, signed-rho 평균 CE와 production-Q CE를 비교한다. 새 head/decoder를 붙이지 않으며 추론과 평가 계약을 유지한다. 각 arm의 모델 유용성과 mean-rho 가설의 상대적 우위는 별도 판정한다. 이는 별도 supervised 실험으로 등록하며, 이번 실패 결과를 대체하지 않는다.

## 원본 및 검증 근거

`results_archive/completed_score_20261016/`에 원본 evaluation, 양쪽 protocol/queue/불변모델 receipt, 독립 검증 및 검증 코드를 보존했다. Score byte SHA는 `860a5d893b2b6e098cf7d5ebaf5718ee124a396319ea357dbad9d77e9c19e3eb`다. 모델/데이터/평가 변경 없이 고친 초기 queue 오류도 `evaluation_recovered_20261016/`에 별도 보존했다. 이번 cycle의 동료 브랜치 fetch는 기존 `cab6c676e24a149a45ea960586c04e579f78d384`와 같았으며 새 성공 결과로 취급하지 않았다.
