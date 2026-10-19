# 동료 coverage 코드 검토: a1df1c3

동료 `origin/patch_v2`의 a1df1c3c2d43693050ff9a55e1895abd83461de1을
확인했다. 변경된 0112 README, evaluate_model.py, test_coverage.py와 STATUS를
읽었다. 이 기록은 새 코드의 CPU 동작 확인이며 실제 모델 성능 결과가 아니다.
우리 237·238 실행 코드와 현재 최고 모델은 변경하지 않았다.

## 재현한 내용

해당 commit의 evaluate_model.py에서 AST로 `_coverage_groups` 함수만 추출했다.
이 함수로 임시 evaluate_model 모듈을 만들고 같은 commit의 test_coverage.py를
그대로 실행했다. Windows CPU에서 동료의 검사 8개가 모두 통과했다.
전체 평가 모듈을 실행하거나 dataset/checkpoint를 읽은 검사는 아니다.
원문 소스와 검사 파일의 SHA는 `coverage_a1df1c3_reproduction.json`에 기록했다.

기존 그룹이 차지하지 않은 patch 중 시간 평균 activity가 이미지별 quantile보다
엄격히 큰 patch만 선택한다. 선택된 patch끼리 4-neighbour 연결 성분을 만들고
최소 크기를 충족하면 새 그룹으로 추가한다. 기존 그룹의 membership은 유지된다.
선택적으로 affinity 하한도 적용한다. ground truth나 object 개수는 이 함수의
입력이 아니다. 실제 평가에는 validation에서 정한 quantile이 사용될 예정이다.

추가 검사에서 기존 그룹 옆의 미할당 patch도 min_size=1이면 별도 그룹으로
추가됨을 확인했다. 따라서 README의 '인접 patch는 미할당으로 남는다'는 문장을
그대로 적용하면 안 된다. 구현이 보장하는 것은 기존 그룹에 편입하지 않는다는
것이다. 같은 물체의 경계 조각이 새 instance로 분리될 위험은 실제 mask에서
평가해야 한다.

## 우리 모델에 적용할 때 확인할 조건

동료는 native32·contract-v2이고 이번 코드는 아직 실제 GPU 평가 전이다.
합성 격자의 세 번째 블록을 복구한 것은 실제 누락 물체 복구 성능의 증거가 아니다.
우리 native64 점수표에 옮길 수 있는 새 점수는 없다.

SW0207과 SW0234는 학습·GT 사용 조건도 다르므로 '새 그룹 생성 여부 하나만
다른 통제 비교'가 아니다. GT가 지정한 merge에서 Object IoU가 하락했다는
사실도 모든 linkage나 pair selection 방식의 가능성을 배제하지 않는다.
이 구분은 이전 `REVIEW_1eced36_KO.md`의 해석을 그대로 유지한다.

activity quantile은 활동량이 누락 물체와 배경을 구분한다는 가정이 필요하다.
엄격한 `>`는 일정한 배경값을 그대로 포함하는 합성 예제의 오류를 막지만,
연속적인 배경 noise나 동일 activity의 물체까지 해결하는 일반 보장은 아니다.
현재 238번의 실제 spike/event/gate 관찰을 먼저 확인해야 한다.
gate와 spike가 같다는 관찰이 나오더라도 gate 활동량 자체의 공간 정보가
없다는 뜻은 아니며, 인과적 gating 기여도 입증되지 않는다.

추후 비교 후보로 채택하면 현재 고정 checkpoint와 동일 image IDs에서
기존 mask와 새 그룹 추가 mask를 짝지어 비교한다. FG-ARI뿐 아니라 두 IoU,
추가 그룹 수와 foreground 증가도 기록해야 한다. 기존 instance의 membership
보존만으로 정확도 향상을 주장하지 않는다. 진행 중인 237번의 확정 평가를
수정하거나 중간 결과에 맞춰 설정을 바꾸지 않는다.
