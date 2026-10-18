# 완료된 mask의 posthoc 진단: grouping을 바꿀 여지가 있는가

**아래 GT-assisted reference는 실제 모델 예측이나 달성한 성능이 아니다.**
기존 예측의 전경/배경 membership을 고정하고, 전경 patch를 GT 물체별로
묶는 구체적인 진단용 partition을 만들었다. 특정 classifier가 이를 학습할
수 있다는 보장이나 새 .90 이상 모델 결과로 해석하지 않는다.

| 모델과 평가 범위 | 실제 FG-ARI | GT로 전경 묶기만 수정한 reference | 놓친 GT 전경 patch 비율 |
|---|---:|---:|---:|
| warm192 seed0 전체320 | 0.877709 | 0.956939 | 6.37% |
| warm192 seed0 공통16 | 0.924401 | 0.959205 | 6.16% |
| warm205 seed1 공통16 | 0.815856 | 0.936003 | 8.69% |
| warm205 seed2 공통16 | 0.788414 | 0.976234 | 3.17% |

특히 seed2 공통16에서 전경 안의 same-object split rate는.091640,
different-object merge rate는.118190이다. 전경 누락만 GT로 보완하는
별도 attachment reference는.799047로, 현재 .788414보다 조금 높다.
반면 membership을 고정한 GT-pure grouping reference는.976234다.
이는 classifier의 split/merge를 다루는 controlled test를 진행할 근거이며,
GT가 없는 실제 예측에서 이 개선이 가능하다는 증거는 아니다.

두 reference는 서로 다른 GT-assisted 조작이며 효과를 더해 합산하지 않는다.
이 보고서는 위의 구체적 reference를 계산했을 뿐 새 전역 optimum/ceiling
정리를 증명하지 않는다. 모든 source masks/GT modal/metric은 독립 검증했고,
전경 고정 reference의 foreground IoU가 원본과 정확히 같음을 검사했다.
추가 model forward/optimizer update는 모두0이다. 실행 중인209/210의
source, model, readout과 사전 판정은 바꾸지 않았다.

원본: `results_archive/support_diagnosis_v1/summary.json`.
재현: `python -m collaborative_test.SW_0208_native64_full320_confirmation.diagnose_fixed_support`.
