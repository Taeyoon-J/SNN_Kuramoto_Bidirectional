# 235·236 이후 실행 판단

이 문서는 결과를 대체하지 않는다. 진행 중인 실험의 검증 결과가 나온 뒤
불필요한 sweep 없이 다음 실행을 선택하기 위한 판단표다. 현재 최고 모델은
SW0227의 고정320장·seed0/1/2 평균 .858916/.605286/.674313으로 유지한다.
고정320장의 FG ≥ .90과 동일 계약 Slot Attention 세 평균 지표 초과가 목표다.

## 235: 학습량 증가가 더 넓은 평가에서도 유지되는가

실제 SW0232 누적512회 모델을 새 학습 없이 기존 first64 validation에서 확인한다.
원래16장 gate 실패는 보존한다. 비교는 같은64장의 원래 모델 및 실제128회 모델이다.
독립 raw-mask 재채점, checkpoint/source binding, 재현 검증을 통과한 결과만 사용한다.

- 등록된 FG 후보 및 balanced gate 모두 통과하면 같은 실제 checkpoint의 전체320장
  평가를 준비한다. 기존64장의 prediction/protocol 재사용 가능성을 검증하고, 나머지
  256장만 추가 추론한다. 새 optimizer update나 checkpoint 재선택은 하지 않는다.
- FG 후보만 통과하고 balanced가 실패하면 IoU 하락을 명시한다. 세 지표의 개선을
  목표로 하는 incumbent 승격이나 바로70,000장 학습으로 건너뛰지 않는다.
- FG 후보가 실패하면 동일 방향의 epoch·LR·계수 sweep을 추가하지 않는다.
  확장64장의 실패 유형을 실제 mask로 분석하고 teacher 최적화와 readout의 불일치를
  검토한다. 기존 성공 모델을 폐기하거나 mini 최고 seed만 선택하지 않는다.

## 236: 데이터 증가 효과가 충분한 업데이트에서 나타나는가

TRAIN64와 TRAIN512 모두 누적512회, batch16, 8,192 exposures의 같은 예산이다.
실제 accepted128 checkpoint/Adam/history/RNG 복원과 새384회 업데이트를 검증한다.
최종 mask는 기존 고정16장·세 seed로 평가한다. encoder와 실제 gamma projection을
학습하며 나머지 core/graph/gates/readout은 유지한다.

- `data_scaling_FG_CI_positive=true`이고 등록된 balanced gate도 통과하면 데이터 증가가
  이 cohort에서 유리하다는 제한된 증거로 보고, 해당 TRAIN512 모델을 first64에
  확장해 확인한다. 즉시70,000장으로 확대하지 않는다.
- FG 평균만 높고 CI가0을 포함하면 데이터 증가 효과를 확정하지 않는다. 235 결과와
  함께 추가64장 확인의 필요성을 판단하며 평가한 모델 중 임의로 checkpoint를 고르지 않는다.
- TRAIN512가 TRAIN64보다 낮거나 IoU 균형이 깨지면 데이터 자체를 쉬운 분포로
  바꾸지 않는다. 동일 teacher에 대한 TRAIN/DEV frequency loss와 최종 spike-derived
  split/merge·foreground precision/recall을 연결해 분석한다.

## 두 비교로 충분히 개선되지 않는 경우

SW0229에서는 split 감소와 소폭 merge 증가가 함께 관찰됐고, foreground recall이
높아도 precision은 낮았다. SW0234의 GT 보조 변환에서는 조각 merge만으로는 Object
IoU가 하락했다. 따라서 foreground 범위를 무조건 늘리거나 모든 조각을 더 합치는
방법을 재시도할 근거는 없다. 실패한 SW0214–216 coarse-merger와 SW0228/230
중심 보존 loss도 계수만 바꾸어 반복하지 않는다.

다음 설계 변경은 실제 실패 mask에서 foreground/background 경계와 물체 사이의
spike assignment 오류를 구분한 뒤 선택한다. TRAIN GT 사용 여부를 명시하고,
DEV GT를 inference나 graph 생성에 넣지 않는다. 변경한 gating의 기여는 별도
제거 비교로 확인하며, frozen parameter 또는 계산 경로의 존재만으로 인과성을
주장하지 않는다. GT 보조 .911458은 새 모델 성능이나 최적 상한이 아니다.

어느 분기에서도 mini 평균으로 전체320장 최고 기록을 덮어쓰지 않는다. 추가 실험은
시작 전 코드·경로·실제 checkpoint 복원과 필요한 smoke check를 완료하고 기존 GPU
exclusive queue에 등록한다. 다른 사용자의 작업을 중단하거나 GPU를 공유하지 않는다.

근거: SW0227·0229·0230·0232·0234의 `RESULT_KO.md` 및 SW0235·0236의
`DESIGN_KO.md`, `completion_audit.py`. 현재235·236의 새 평균 결과는 미확정이다.
