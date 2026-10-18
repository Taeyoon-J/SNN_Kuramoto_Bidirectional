# SW0215: loss/판정 보정은 하락을 완화했지만 baseline 개선 없음

Native64·고정16·seed2, 기존214 head를 재학습 없이 사용했다. Local4/Linux4
테스트 후 CPU에서214의 두 원래 endpoint를 모두 bitwise 재현하고 새 mask를
고정한 뒤 GT를 읽었다. 모든80 mask endpoint, 세 지표·paired CI와 source
component/전경 보존을 독립 재검증했다.

| 설정 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 core | .788414 | .445474 | .674335 |
| 214 raw primary | .720607 | .445474 | .580461 |
| **보정 primary** | **.781651** | **.445474** | **.664934** |
| 214 raw frozen geometry | .814008 | .445474 | .632410 |
| 보정 frozen geometry | .788414 | .445474 | .674335 |

TRAIN의 image-balanced positive mass1.178666, negative mass30.821334,
positive prior .0368333에서 log(N/P)=3.263823475를 계산해 raw logit에서 뺐다.
Validation 점수를 이용해 threshold를 선택하지 않았다. 이상적 weighted BCE
해의 log-odds shift를 맞추는 것으로, 유한 classifier 확률의 calibration을
보장하지 않는다.

Primary는 raw214보다 나아졌지만 원래 core 대비 FG 차이-.006763,
95% CI[-.018548,+.000049]로 개선은 확인되지 않았다. 평균 merge는9.3125에서
1.9375로 줄었다. 보정 geometry control은 merge0으로 원래 mask와 같다.
.8140 결과를 유지하기 위해 임의의 중간 threshold를 찾지 않는다.

다음에 점검할 중요한 target 차이도 있다.214의 object histogram은 component
전체 patch 수(정답 background 포함)로 나누지만 FG-ARI는 정답 foreground
patch에서 계산한다. 따라서 background가 섞인 같은 물체 component의 학습
target이 낮아질 수 있다. 후속 시험은 같은 데이터·평가 계약을 유지하면서
TRAIN 관계 target을 foreground-conditional로 정의하는 방안을 먼저 검토한다.
Anchor 순서 불변 특징은 별도 가설이며 이 결과로 원인이라고 단정하지 않는다.

새 최고 성능은 없다. 전체320·세 seed 기준점 .848615/.592533/.672047을
보존한다. Core forward0/추가 optimizer update0이며 전체70k 실험이 아니다.
증빙: results_archive/completed_v1/score_v1/summary.json,
predictions/protocol.json 및 completion_audit.json.
