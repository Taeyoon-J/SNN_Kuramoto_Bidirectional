# SW0260: 고정32 core + 학습 가능한64 readout 최적화

이번 비교에서 가장 좋은 검증 설정은 **context_rgb**다. 기존 SW0259를 보존하고,
원래 encoder·core는 고정한 채64×64 readout의 loss·구조를 비교했다.
아래 비교는 모두 CLEVR 고정320장(ID1320–1639), **동일64×64 정답 patch**, seed0/1/2다.
과거32×32 점수0.821628과 이64×64 점수를 직접 비교하지 않는다.

## 최종 비교

| 모델 | FG-ARI | Raw Foreground IoU† | Matched Object IoU |
|---|---:|---:|---:|
| 기존259 mask를64로 펼쳐 동일 기준 재평가 | 0.802087 | 0.211509 | 0.655888 |
| 학습 전 새64 구조 | 0.797807 | 0.211509 | 0.659454 |
| **학습한 context_rgb** | **0.806885** | 0.211509 | **0.670983** |
| 학습한 fine CNN branch를 끔 | 0.798240 | 0.211509 | 0.655236 |
| 입력 gamma의2×2 세부 차이를 평균으로 제거 | 0.801269 | 0.211509 | 0.663821 |

| seed | 기존259의 동일64 FG-ARI | 새 모델 FG-ARI | 새 모델 Object IoU |
|---|---:|---:|---:|
|0|0.837784|**0.847099**|0.720752|
|1|0.777612|0.776843|0.645283|
|2|0.790865|**0.796712**|0.646914|
|평균|0.802087|**0.806885**|0.670983|

기존 동일64 기준 대비 평균FG-ARI 차이는 **+0.004797**, 학습 전 구조 대비는
**+0.009078**다. Fine CNN을 끄면 **−0.008644**, gamma를 거칠게 만들면 **−0.005616**이다.
기존259 실험과 달리, 학습된 head/fine branch가 최종 mask에 기여한다.
모든 seed가 개선된 것은 아니다: seed1은 약0.000769 낮다.

동일한 세 seed를 고정하고320개 이미지만 함께 bootstrap5000회 재표집한 개선량95%구간은
**[+0.002735,+0.006719]**다. 이는 이미지 표본 불확실성에 관한 값이며, seed 모집단의
불확실성이나 새로운 데이터셋 일반화를 입증하지 않는다. 이320장은 역사적으로
반복 사용한 연구 평가 집합이며, 새로 보지 않은 TEST로 주장하지 않는다.

†모든 예측 그룹을 양수ID로 유지하고 배경을 지정하지 않는다. 따라서 raw Foreground
IoU는 정답 전경 면적 비율이다. 배경을 따로 판별한 의미상 전경 검출 IoU가 아니다.
학습용 정답 mask는 사용하지 않았지만, **검증 mask는 모델·epoch 선택에 사용했다.**

## 실제 구조와 학습 범위

```text
RGB128 → 기존 고정 encoder → gamma32 → 기존 고정 core32
                                            ↓
                              실제 spike·membrane 특징144D
                                            ↓ 32→64 공간 확장
같은 고정 encoder → 실제 gamma64 → 새64-grid CNN → head embedding에 결합
                                            ↓
                       후보 그룹별 점수 →64×64 최종 mask
```

- Core는1024개 위치(32×32),1024step·settle512 그대로다.4096개 독립 oscillator가 아니다.
- 기존 encoder와 core 가중치를 변경하거나 core로 역전파하지 않았다. Core 활동은 캐시했다.
- 새로 학습하는 fine CNN은 gamma64의8채널에3×3 Conv8→16,SiLU,3×3 Conv16→32를 적용한다.
  마지막 convolution은0으로 초기화한다. 새 CNN은 head 쪽에 있으며 원래 encoder와 구분한다.
- 후보 점수를 지역적으로 비교하고, 고정된 RGB content 표현을 사용해 head embedding이
  grouping을 거치지 않고 RGB를 표현하는 직접 gradient 경로를 제거했다.
- 최종 선택 모델은 **기본 RGB reconstruction MSE**를 사용하고 좌표를 사용하는 공유
  decoder를 유지한다. 특수 loss를 더한 모델이 최고가 된 것은 아니다.
-64×64 gamma와 CNN을 실제 계산하므로 최종 mask를 단순히4배 복제한 모델이 아니다.
  세부 gamma와 학습한 CNN을 제거한 비교도 위 표에 포함했다.

## 실험한 조건과 결과

| 단계 | 조건 | 검증32장 평균 FG-ARI | 고정320장 FG-ARI |
|---|---|---:|---:|
|기준|기존259를 동일64로 평가|0.793321|0.802087|
|R1|RGB /TV /spike /proposal anchor /반경 변경, 상위2조건 확대|최고0.804811|선택RGB0.802209|
|R2|초기259 mask를 정확히 보존하는 bounded delta 세 조건|최고0.789735|검증 미개선으로 평가·확대 생략|
|R3|decoder 좌표 제거 /경계 가중 RGB /좌표+경계 가중|최고0.806348|선택flat_edge0.801285|
|R4|fine CNN /fine CNN+flat edge /넓은 receptive field|**최고0.811091**|**선택context_rgb0.806885**|

14개 조건을 각3seed에서 짧게 학습해42개 screen을 완료했고, 유망한12개 경우를
추가 학습했다. 같은 조건·seed의 확대 학습은 별개 독립 모델 수로 과장하지 않는다.
원래 모델·각 negative arm·선택 근거·checkpoint는 보존했다. 조건 선택은 별도 검증
1000–1031만 사용했고, 최종320장은 선택한 예측을 고정한 뒤 평가했다.

기본 screen은 고유 TRAIN128장×8회(1024Adam updates)이며 R2만10회(1280updates)였다.
유망한 경우 고유 TRAIN256장×20회(5120updates)를 추가했다. 같은 데이터의 부분집합이므로
고유 이미지가128+256장이 되는 것은 아니다. 확대 모델의 실제 Adam step은6144다.
학습률은 screen.001→확대.0003, clipping1, physical batch1이다.
출발점은 기존259의 mask-free RGB best head·decoder이며, 그 이전 학습 이력도 상속한다.
35,000장 또는70,000장으로 이번 readout을 새로 학습한 결과가 아니다.

검증32장으로 여러 조건을 선택했으므로 검증 과적합 가능성은 남는다. R3는 검증에서는
더 높아도 고정 평가에서는 개선되지 않았다. 현재 결과를 loss의 전역 최적해, 전체 gating의
인과적 기여, 학습량 증가 효과 또는 end-to-end core 학습 효과로 확대 해석하지 않는다.

## 검증·보존

최종 각 모델에서320장×3seed×5비교=4800개 mask를 독립 NumPy ARI/Hungarian IoU로
재계산했다. 실제 Adam step·학습 로그·frozen content·원래 core/encoder checkpoint SHA·
과학 소스·분할·예측 SHA를 검사하고, 별도 CPU 예측9장을 정확히 재현했다.
추가로 새 Python 프로세스에서 세 seed 각각의 전체 이미지→64mask 추론을 재현했다.

[최종 결과](results/context/final_results.json),
[독립 검증·bootstrap](results/context/independent_audit.json),
[초기64 비교](results/r1/final_results.json),
[R2 negative 결과](results/delta/screen_results.json),
[R3 결과](results/objective/final_results.json),
[checkpoint 안내](CHECKPOINTS_KO.md).

서버 최종 모델: `/work/USERS/kevinswk/experiments/SW0260_context64/extended/seed{s}/context_rgb/best.pt`.
기존 공개 SW0259 source·checkpoint는 그대로 유지한다.
