# SW0259: SW0165 기반 비지도 readout 비교 결과

2026-11-01 America/New_York. 모든 수치는 동일 CLEVR 고정320장(ID1320–1639),
native32×32 patch 평가이며 seed0/1/2를 사용했다. 이 고정 평가 집합은 역사적으로
반복 사용한 기준점이며, 새로운 미사용 TEST 성능으로 해석하지 않는다.

## 결과

| 구성 | FG-ARI | Foreground IoU | Matched Object IoU |
|---|---:|---:|---:|
| 기존 SW0165, 세 seed 평균 | 0.804299 | 0.609975 | 0.651149 |
| SW0259 RGB 복원 학습, 세 seed 평균 | **0.821628** | 0.218430† | **0.716835** |
| R1 기존 영역 prior 완화 + 활동 일관성, 세 seed 평균 | 0.740369 | 0.218430† | 0.674830 |

| seed | SW0165 FG-ARI | SW0259 FG-ARI | SW0259 Foreground IoU | SW0259 Object IoU |
|---|---:|---:|---:|---:|
| 0 | 0.846399 | **0.859454** | 0.218430† | 0.759579 |
| 1 | 0.770932 | **0.796401** | 0.218430† | 0.699946 |
| 2 | 0.795566 | **0.809030** | 0.218430† | 0.690978 |

SW0259는 세 seed 모두 기존 FG-ARI보다 높고 평균 차이는 +0.017329다.
Object IoU 평균 차이는 +0.065686이다. Seed 모집단의 통계적 유의성이나
새 데이터셋 일반화까지 입증한 결과는 아니다.

†새 모델은 모든 그룹을 양수 ID로 유지하고 배경을 지정·삭제하지 않는다.
따라서 보호 평가 함수의 `prediction != 0`은 이미지 전체가 되며, raw Foreground
IoU는 정답 전경 면적 비율이다. SW0165의 의미상 전경 검출 IoU와 직접 비교하지 않는다.
FG-ARI는 정답 전경 위치에서 계산하며 예측 그룹 번호에는 의존하지 않는다.

## 무엇을 유지하고 학습했는가

- 같은 seed의 SW0165/SW0097 source core·encoder·정규화 통계, strict16→32 mapping,
  graph-T,1024 step·settle512를 그대로 고정했다. Encoder와 core의 새 학습은 없다.
-252의 NeutralRegionHead 구조를 그대로 사용하되 가중치를 새로 초기화했다.
 252의 정답 mask로 학습된 head·encoder 가중치를 사용하지 않았다.
- Head 입력은 patch별 실제 spike·membrane144차원 특징이다. 후보 영역은 이미지 전체의
  signed partition 및 원래 head의12개 특징 기반 후보다. 가장 큰 그룹을 배경으로
  제거하지 않는다. SW0165 baseline mask는 별도 경로로 그대로 재현해 비교했다.
- 공유 RGB decoder를 추가했다. 각 그룹의 pooled32차원 표현과 좌표만으로 RGB를
  복원하고 soft mask로 합성한다. 입력 RGB는 복원 정답으로만 사용한다.

학습은 동일 TRAIN256장×20epoch(각 arm5120 Adam updates), 검증은 별도 RGB32장
(ID1000–1031)이다. TRAIN은1000–1639를 제외한 등록된70000 pool에서 RandomState259로
선정했다. 세 seed에 같은 이미지를 사용했다. 물리 batch1, Adam.003, clip1이다.
실제 최종 epoch 선택은 RGB reconstruction validation MSE로 했다: 기본 RGB arm은
seed0/1 epoch20, seed2 epoch18이 선택됐다. 70000장으로 새 학습한 결과가 아니다.

첫 라운드3개 loss 조건은 RGB / RGB+활동 일관성(.01) / 앞 조건+entropy(.001)였다.
세 seed의 평균 RGB validation MSE로 공통 RGB arm을 선택한 다음,320장 예측을 고정하고
평가 정답을 열었다. 최종 FG-ARI를 보고 seed별 다른 arm을 고르지 않았다.

## 반드시 함께 보고할 한계: 높은 점수의 원인

**최고 SW0259의 최종 hard mask는 학습 전 후보 영역과 전체960장 모두 정확히 같았다.**
Head·decoder 가중치와 soft mask·복원은 학습됐지만, 기존 영역을 유지하는 prior가 강해
최종 argmax 배정은 바뀌지 않았다. 세 seed의 `prior_scale > 2*abs(residual_scale)`도
확인했다. 따라서0.804299→0.821628 향상을 **RGB loss가 학습한 분할 개선**이라고
주장할 수 없다. 향상은 전체 영역 후보 생성과 배경 처리 변경에서 나온다.

정답 없는 검증 이미지에서 이 문제를 먼저 진단하고 R1을 별도 실행했다. 원래 실험은
유지했다. R1은 각 seed의 mask-free RGB 학습 checkpoint에서 출발해 prior를.5로 낮추고
초기 residual을 최소2로 설정했다. Prior는[0,.5]로 제한하고.003 L2를 적용했다.
RGB만 사용하는 조건과 활동 일관성(.03)을 더한 조건을 각각256장×20회 추가 학습했다.
추가 Adam update는 각5120회이며, 이전 best epoch까지의 학습 이력도 따로 보존했다.

R1의 공통 조건은 역시 RGB 검증 오차로 선택했다. 실제 patch 배정은 움직였지만
FG-ARI는0.740369로 하락했다(seed별0.804291 /0.679365 /0.737451).
**복원 오차 감소와 물체별 grouping 개선이 일치하지 않는 문제는 남아 있다.**
기존 후보의 좋은 동기화 정보를 보존하면서 학습된 head가 유용한 수정을 만드는 것은
후속 연구 과제다. 이번 결과를 gating 기여나 학습된 head의 효과 입증으로 확대하지 않는다.

## 검증과 저장 위치

- 원래 SW0165의320장×3seed mask를 모두 bitwise 재현했다.
-1920개(두 라운드×320장×3seed) mask의 지표를 독립 NumPy ARI/Hungarian IoU로 재계산했다.
- 각 라운드·seed에서 별도 CPU 예측3장을 재현했고, 예측 SHA·TRAIN IDs·실제 Adam5120 steps·
  20epoch 학습 로그·40개 과학 소스 해시·고정 core·보호 평가 코드12개를 검증했다.
- Core 활동은 seed/input/source에 묶인 캐시를 재사용했다. GPU3에서 seed2 평가 활동을
  별도로 생성해 동일 출처를 확인한 뒤 연결했다. 학습3seed와 평가 준비를 겹쳤다.
- 원래 큐 시작부터 첫 라운드320장×3seed 평가 완료까지 약53분24초였다.

서버: `/work/USERS/kevinswk/experiments/SW0259_unsupervised_readout`.
원래 실행 소스는 `code/c110dda08ae6`이며 실행 후 수정하지 않았다.
최고 head+decoder는 `seed{s}/arms/rgb/best.pt`; 원래 core·encoder는 source 경로 그대로다.
후속 모델은 `refinement_prior_release/seed{s}/release_activity/best.pt`에 보존했다.

검증된 원문: [첫 라운드 결과](results/results.json),
[독립 검증](results/independent_audit.json),
[전체 hard-mask 비교](results/hard_routing_audit.json),
[실제 source·Adam 검증](results/sources_optimizer_audit.json),
[R1 결과](results/refinement_prior_release/results.json),
[R1 독립 검증](results/refinement_prior_release/independent_audit.json).
