# SW0197 결과: native64 기준 모델의 seed 재현

우리 데이터 고정 평가 계약에서 128×128 RGB를64×64 patch로 평가한 **고정32장
development mini** 결과다. native32에서 정답 mask로 추가 학습한186의 세 seed를
native64로 옮긴 기준 모델이며, 추가 native64 학습이나 새70k 학습은 없다.
seed0은193의 baseline이며 TRAIN8 후보(.9186)를 섞지 않았다.

| 모델 | FG-ARI | Foreground IoU | Matched object IoU |
| --- | ---: | ---: | ---: |
| 우리 seed0 | .909537 | .827147 | .734442 |
| 우리 seed1 | .817512 | .447770 | .666477 |
| 우리 seed2 | .827236 | .459299 | .681092 |
| 우리 세 seed 평균 | .851428 | .578072 | .694004 |
| 같은32장 Slot 세 seed 평균 | .821571 | .173315 | .203988 |

실제 우리96 mask, Slot96 mask와 canonical2×2 modal GT를 재검증했다. 생산 metric과
독립 계산 차이는 최대2.22e-16. 세 metric의 mini 평균은 모두 높다. FG 평균 차이는
+.029857, 고정 모델에서 이미지32개 paired bootstrap CI[-.026523,.082492]다.
이 표본으로 FG 우위가 확실해졌다고 주장하지 않는다. 최종320장 평가와 세 seed
평균.90 목표는 아직 입증되지 않았다. 우리 모델은 mask-supervised 추가 학습
모델이므로 unsupervised leaderboard 결과로 등록하지 않는다.

## 약한 seed의 오류

동일한 frozen mask에서 계산한 조건부 오류율이다. 분모가 서로 다르므로 합쳐서
성능 하락의 기여도 백분율로 해석하지 않는다. 새 forward/학습은0회다.

| 진단 평균 | seed0 | seed1 | seed2 |
| --- | ---: | ---: | ---: |
| 실제 foreground를 놓치는 비율 | .052574 | .070077 | .023133 |
| 실제 background를 foreground로 잡는 비율 | .055053 | .305521 | .313141 |
| 같은 실제 물체 patch pair를 나누는 비율 | .060621 | .139870 | .105742 |
| 서로 다른 실제 물체 pair를 합치는 비율 | .024432 | .038811 | .056925 |
| 예측 foreground precision | .868681 | .460873 | .465950 |
| 생성 mask의 양성 label 수 | 8.03125 | 15.53125 | 18.00000 |

실제 물체 label 수 평균은5.65625다. 표의 mask 수는 양성 label 개수 진단값이다.
seed1/2의 background 오검출 증가는 seed0 대비+.2505/+.2581이고 paired CI도
양수다. 같은 물체를 나누는 오류 증가도 두 seed에서 CI가 양수다. 학습 seed
집단 전체의 분산이나 인과 원인을 증명한 결과는 아니다.

Foreground IoU 개선에는 배경 과검출을 줄여야 한다. FG-ARI는 실제 foreground의
instance partition을 평가하므로 foreground 내부 분할·병합 오류를 줄이는 학습이
필요하다.199의 절대 연결 calibration과200의 settled-window gradient 확장이
이 문제를 개선하는지 먼저 mini로 확인한다. seed0의 좋은 결과만으로 최종 세
seed 성공을 선언하지 않으며, 유망한 방법은 seed1/2에서도 재현해야 한다.

## 동료의 새 group-count 방향 검토

동료 commit1fc3906의95/96은 group-size floor/graph density 변경으로 과분할을
해결하지 못했고, 다음 단계로 raw PLV participation 목표7.5를 제안했다. 아직
그 count-loss 학습 결과는 공개되지 않았다. 서로 다른 평가 계약의 수치를
직접 비교하거나 그 목표값을 우리 loss에 그대로 넣지 않는다.

실제 로컬 `plv_group_count_loss`와 문서의 raw PR 공식은 이상적인 크기
[90,5,5]의3 block에도1.22699를 출력하고 target3에서 loss .34928을 준다.
등록된 TRAIN64의 완벽한 GT block affinity도 실제 class 수 평균7.10938에 대해
raw PR은1.57845, 정답 class 수를 목표로 한 raw loss 평균 .59158이다. 이는 실제
학습 affinity 성능 결과가 아니라 **이상적인 정답 partition의 수학적 점검**이다.

degree-normalized affinity의 participation prototype은 이상적인 불균형 block에서
올바른 block 수를 준다. 작은 CPU 검사5개를 로컬/Linux에서 통과했으나 실제
connected-component 개수나 학습된 count classifier를 보장하지 않는다. CPU
prototype을 검토한 상태이며 모델 학습 성능 개선은 아직 시험하지 않았다.
현재199/200의 source/loss/실행 조건은 이 검토로 바뀌지 않았다.
