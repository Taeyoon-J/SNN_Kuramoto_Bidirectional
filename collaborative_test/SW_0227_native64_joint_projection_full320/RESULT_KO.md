# SW0227: native64 전체320장 최고 FG 기준점 갱신

고정 320장(IDs1320–1639), native64 modal patch GT, seed0·1·2 평균이다. SW0223의 고정 checkpoint를 사용했고 추가 학습은 없다. 기존 64장 mask를 재사용하고 seed별 256장을 추가 생성했다. 실제 source checkpoint와 mask/frequency 재현 검증을 통과했다.

| 모델 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 이전 기준 SW0210 | .848615 | .592533 | .672047 |
| SW0227: encoder + gamma projection 학습 모델 | **.858916** | **.605286** | **.674313** |
| 같은 계약의 Slot Attention | .786902 | .198201 | .195203 |

이전 기준 대비 FG 차이는 +.010301, paired image bootstrap 95% CI [.005659,.014751]이다. Foreground IoU 차이는 +.012753, CI [.001770,.023733]이다. Object IoU 차이는 +.002266, CI [-.002547,.006817]이므로 Object IoU 개선은 아직 확정하지 않는다. CI는 고정 세 seed에 조건부인 image bootstrap이며 seed 모집단 불확실성은 아니다.

| seed | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 0 | .882964 | .804963 | .693819 |
| 1 | .839643 | .529764 | .667478 |
| 2 | .854141 | .481131 | .661643 |

원시 mask 2,880개(candidate·이전 기준·Slot)를 독립 재계산했다. 집계·CI 오차는 5.56e-16 이하이고, 세 지표 평균 모두 Slot을 엄격히 초과한다. 각 seed가 개별적으로 FG에서 Slot을 이긴다는 뜻은 아니다. mini16에서 실패했던 기존 gate 기록은 보존하며 full320 근거와 구분한다.

이 결과를 고정 전체 평가의 새 FG 기준점으로 등록한다. FG .90 목표는 미달성이다. mask-supervised lineage이며, 새 native64 70,000장 학습 결과나 동일 supervision/cost의 unsupervised 비교가 아니다. 반복 사용한 validation으로, untouched test 성능이라고 부르지 않는다. 모든 gating의 기여나 데이터 증가 효과는 이 결과만으로 입증되지 않는다.

232번은 이 실제 checkpoint에서 추가 학습이 효과가 있는지 별도로 검증 중이다. 원본 최고 모델과 이번 checkpoint는 모두 보존한다.
