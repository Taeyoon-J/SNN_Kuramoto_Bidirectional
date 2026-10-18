# SW0216: FG-conditional target 변경도 개선 실패

동일 source205 seed2/cache/TRAIN32/고정16/native64에서214와 feature table,
image-pair weight, MLP 초기화가 정확히 같음을 검증했다. TRAIN target의
분모만 정답 foreground 수로 바꾸고, 새 TRAIN 통계로 판정 보정을 계산했다.
Local17/Linux17 테스트와 실제6648-pair gradient/초기 readout 사전검증 후
GPU에서 Adam128을 완료했다.64개 mask endpoint와GT-assisted 참고값까지
독립 재검증했다. 데이터나 세 평가 metric의 정의는 바꾸지 않았다.

| 설정 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 core | .788414 | .445474 | .674335 |
| matched all-patch target + 보정215 | .781651 | .445474 | .664934 |
| **FG-conditional target + 보정216** | **.781651** | **.445474** | **.665117** |
| frozen geometry control | .788414 | .445474 | .674335 |

FG 차이는 기존 대비 -.006763, paired95% CI[-.018548,+.000049]다.
Matched215 대비 FG 차이는0이다. TRAIN positive prior는 .036833→.048753,
log-odds shift는3.263823→2.971002로 바뀌었으나 유효한 개선은 없었다.
Primary의 평균 merge는2개, 예측 group19.8125개, 정답 평균5.375개였다.

정답을 알고 같은 complete-link 규칙으로 합친 **참고 partition**은
all-patch target에서 FG .847321, FG-conditional target에서 .849997이었다.
이는 실제 모델 점수도, 인증된 최적값/학습 가능한 상한도 아니다. 다른
merger가 이보다 높을 가능성을 배제하지 않는다. 다만 이 예시와 실제 결과는
현재 target 분모 변경만 반복할 근거를 주지 않는다.

Merge-only 접근은 원래 한 component에 섞인 여러 물체를 나눌 수 없다.
또한 현 native solver의 각 그룹은 이미4-connected이므로 단순 connected
component 분리는 identity이며 새 테스트로 반복하지 않는다. 다음은 실제
spike Q와 coupling A/gain의 기존 signed energy에서 잘못된 merge를 되돌릴
수 있는지 확인하는 분할 보정 방향이다. 기존 energy와 source foreground를
유지하고 새 판정은 GT 없이 수행해야 한다.

현재 전체320·세 seed 기준점 .848615/.592533/.672047을 유지하며 .90 목표는
미달이다. 추가 core forward/encoder update0, supervised classifier pilot이다.
증빙: results_archive/completed_v1/score_v1/summary.json,
head_training/report.json, completion_audit.json 및 deployment_v1의17-test/admission
영수증. Feature/weight/initial-model equality와 실제 CUDA Adam128을 검증했다.
