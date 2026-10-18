# SW0224: cell 정렬만으로 FG 개선은 확인되지 않음

CNN/core/source 통계/128 RGB/2px modal GT/ID/readout을 고정하고126 feature에
복제 경계 한 줄을 붙여64 pooling cell을 정렬했다. 추가 학습0이다. 같은 source
세 모델과DEV16에서 비교했고 세 protocol 모두 동결한 뒤 GT를 읽었다.

| 세 seed 평균 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 기존 source | .842891 | .600692 | .702161 |
| cell 정렬 | .838302 | .586019 | .713245 |
| 차이 | -.004588 | -.014672 | +.011084 |

FG paired image95% CI[-.035006,.037387], Foreground IoU[-.051109,.014607],
Object IoU[-.022370,.044649]로 모두0을 포함한다. FG/balanced 확대 gate는 실패했다.
Object IoU track 후보는 보존하지만 확실한 개선이나 새 최고 모델로 승격하지 않는다.
FG 우선순위에 따라 이 무학습 adapter의 확대·padding/stride sweep은 하지 않는다.

축당 평균 feature cell 중심 오차 .5px -> .015625px라는 기하 확인과,
세 seed 평균 segmentation 성능 개선 여부는 다른 질문이었다. 첫 두 seed의FG는
하락했고 seed2는 상승했다. 이 결과만으로 모든 정렬 방법이나 학습 후 효과까지
부정하거나, 원래 성능 문제의 원인이 pooling 하나였다고 주장하지 않는다.

원본 checkpoint byte SHA와 모델 parameter SHA/통계 유지, adapter 코드 SHA,
96개 raw mask의 독립 metric 및 모든 평균·paired CI를 감사했다. 최대 오차는
2.22e-16이며 별도 로컬 평균/CI 재계산 차이는0이다. 감사의 추가 forward/학습0,
RGB/GT resize/crop0. 증거는 results_archive/completed_v1에 있다.

source의 mask-supervised lineage를 상속한다. 추가 학습이 없다는 이유로 이
전체 계열을 unsupervised 모델로 부르지 않는다. native64 전체320 최고
.848615/.592533/.672047과223의16장 결과는 변경하지 않는다.
