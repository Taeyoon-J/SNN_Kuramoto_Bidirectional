# SW0194 완료: native64 TRAIN64 추가 학습

범위:64×64 patch, seed0, 미리 정한 development32장. 아래64장은 기존 foundation의
전체 학습량이 아니라 이번 계열에서 추가 adaptation에 사용한 고유 이미지 수다.
Mask-supervised 학습이며 새70,000장 전체 학습이나 unsupervised 성과가 아니다.

| 모델 | FG-ARI | Foreground IoU | Matched object IoU |
| --- | ---: | ---: | ---: |
| Native64 전이 baseline | .9095369755 | .8271468754 | .7344419700 |
| TRAIN8 추가 학습 | .9185864261 | .8328674431 | .7336862632 |
| TRAIN64 추가 학습 | .9001655085 | .8308344896 | .7316582231 |
| 같은32장 Slot3mean 참고 | .8215711846 | .1733153073 | .2039881566 |

우리 모델의 위 세 행은 seed0 한 개다. Slot3mean 참고 행과 이 표만으로
전체320장/세 seed 평균의 우위를 입증했다고 표시하지 않는다.

TRAIN64−TRAIN8 FG 변화는-.0184209176, paired-image bootstrap95%CI는
[-.0472437519,.0013289966]. IoU 변화는 각각-.0020329535와-.0020280401이다.
세 지표의 paired CI 모두0을 포함한다. 개선이 확인되지 않았고 평균 점수도 낮다.
등록한 FG gain>=.01 및 CI하한>0 조건은 실패했다. 최고 모델을 교체하지 않는다.

이 결과만으로 과적합을 확정할 수 없다.195번은 같은 TRAIN8/TRAIN64 모델의
TRAIN16 및 development32 성능 변화를 비교한다. 데이터와 optimizer 기회가
동시에 증가했으므로194번 자체도 독립적인 causal data-scaling 실험은 아니다.
현재 TRAIN64 recipe는 닫고 작은 LR/weight 조정으로 계속 연장하지 않는다.

실행·검증: 총16 B4/micro1 proposals,64 exposures,12 accepted updates/실제Adam12.
여기서64 exposures는 logical gradient-training 이미지 수다. Proposal마다 같은
이미지의 before/after no-grad forward도 수행한다. 거절된 proposal은3,14,15,16이며,
실제 반영된12개 proposal에 포함된 고유 이미지는48장이다.64장 모두의 update가
반영되었다고 해석하지 않는다.
기존 두 proposal의 model/Adam/CPU·CUDA RNG 및 history를 정확히 이어갔다.
기존 source checkpoint는 그대로 보존했다. 새14 proposal의 기록 시간 합은
6181.84초, candidate32장 추론 합은1073.09초다. 대기·초기화·checkpoint I/O 등은
이 두 합에 포함하지 않는다.192번에서 이미 끝난 두 proposal의 시간도 제외했다.

완료 checkpoint의 실제 model/Adam/history와96개 우리 mask 및96개 Slot mask를
독립 검증했다. 세 지표·평균·CI의 최대 수치 오차는2.220446049250313e-16이다.
공개 기록은 집계와 해시만 포함하며 imagewise GT-derived 점수는 비공개 보관한다.

Summary SHA256:5b3fbfadd86dd4431f34bdb09b9b275bbdf6caa9a02ebb1e8d2fa135129cda13.
최종 model SHA256:43b562b6448438c4843f32030d236922e6f9be256e61ea44d5b9618e89342e65.
