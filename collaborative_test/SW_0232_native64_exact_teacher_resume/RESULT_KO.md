# SW0232: 추가 학습 평균은 상승, 고정16장만으로 FG 개선 확정 불가

**장수와 업데이트 구분:** 기존 checkpoint에 추가 학습한 고유 이미지64장,
해당 추가 학습 단계의 누적 optimizer update512회, 평가 이미지16장,
seed0·1·2 평균 FG .884622다. 모델의 전체 학습 이력이64장뿐이라는 뜻도,
512장으로 학습했다는 뜻도 아니다. 기존 checkpoint의 이전 학습 이력은
그대로 상속한다. [평가 범위 구분](../EVALUATION_SCOPE_KO.md)을 함께 확인한다.

실제 accepted SW0223 128회 checkpoint의 encoder·projection·Adam·history·Torch RNG를 세 seed 모두 bitwise 복원했다. 같은 TRAIN64·teacher·LR·graph·나머지 gate/readout으로 384회 추가해 누적512회 학습했다. 새 노출은6,144회이며 상속된128회를 새 학습으로 세지 않는다.

고정16장·native64·세 seed 평균:

| 모델 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 원래 기준 | .842891 | .600692 | .702161 |
| 실제128회 SW0223 | .859634 | .645611 | .706340 |
| 누적512회 SW0232 | **.884622** | **.677086** | **.712312** |

128회 대비 FG 차이는 +.024988, paired image bootstrap 95% CI [-.006611,.062020]이다. Foreground IoU 차이는 +.031475, CI [.000588,.067281]이다. Object IoU 차이는 +.005971, CI [-.010016,.021859]이다. FG 평균 상승은 유망하지만 고정16장만으로 확정하지 않는다. 원래 FG/full320 candidate gate와 balanced gate는 모두 **false로 보존**한다.

원본128회 prefix·새384회 budget·세 actual512 checkpoint·teacher·raw mask144개를 독립 검증했다. 원시 점수·평균·CI 오차는1e-12 미만이며 로컬 재계산 오차는0이다. 실제 teacher loss와 DEV frequency의 물체 내부 변동은 세 seed 모두 줄었다. 이것만으로 최종 mask 개선이나 모든 gating의 기여를 단정하지 않는다.

다음은 예산이나 loss를 바꾸지 않고 이 고정 checkpoint를 기존 first64 validation에서 확인하는 SW0235다. 이전 SW0223도 동일한16장 불확실성을 secondary64에서 확인했으며, 기존 실패 gate를 덮어쓰지 않았다. 이번에도 같은 원칙을 따른다. 이미 사용한 validation이므로 untouched test로 취급하지 않는다. 실제128회 모델과 원래 모델을 모두 같은64장 조건에서 비교한다.

현재 전체320장 최고는 SW0227의 .858916/.605286/.674313이며 이 미니 결과로 교체하지 않는다. 신규70,000장 학습도, .90 전체 평가 달성도 아니다.
