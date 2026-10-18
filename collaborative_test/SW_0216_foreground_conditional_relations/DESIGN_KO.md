# SW0216: foreground-conditional TRAIN relation target

214 histogram은 정답 background를 포함한 component 전체 크기로 나누었다.
215의 TRAIN-only loss/decision 보정은 큰 하락을 완화했지만 baseline을 넘지
못했다. FG-ARI는 정답 foreground patch에서 계산하므로, 같은 물체의 두
component에 background가 섞였다는 이유로 relation target까지 작아지는
차이를 이번에 바꾼다. 평가 데이터/label rule/세 metric은 바꾸지 않는다.

Source205 seed2, 기존 cache/TRAIN32/DEV16/native64, 동일548 pair 특징,
동일MLP548→64→1/초기seed214002/Adam1e-3/full-batch128/clip1을 유지한다.
Feature table과 image-pair weight의 SHA가214와 정확히 같아야 한다.
오직 TRAIN histogram의 분모를 해당 component의 정답 foreground 수로
바꾼다. Foreground가 없는 component의 histogram은0으로 두며 배경끼리를
물체로 합치지 않는다. Mixed foreground는 soft target을 유지한다.

Positive/negative-normalized BCE와215의 TRAIN-only log(N/P) 보정을 사용한다.
새 target에서 P,N을 다시 계산하고 보정 후 strict logit>0 complete-link를
적용한다. DEV threshold/epoch/LR sweep 없음. Original component 내부와
예측 foreground는 유지한다. RGB/GT count/DEV GT를 inference에 넣지 않는다.

Primary와 frozen geometry-only control을 미리 고정한다. 기존215의 보정된
all-patch target endpoint를 matched 비교로 포함한다. 모든 prediction을
검증한 뒤 GT를 읽는다. 이후 두 GT-assisted component-merging 참고값도
계산하되 모델 점수/학습 가능한 상한/인증된 optimum이라고 부르지 않는다.

Primary FG+.01, paired95% CI lower>0, ObjectIoU 감소 없음이면 확대를 검토한다.
FG만 유용한 후보도 selection policy에 따라 보존한다. 실패하면 target/계수의
작은 sweep을 하지 않고 merge-only 제약과 표현 일반화 문제를 판단한다.
Core forward/update0, supervised pilot이며 전체320/세seed .90 또는새70k
학습을 완료했다고 주장하지 않는다. 기존 .848615 reference는 보존한다.
