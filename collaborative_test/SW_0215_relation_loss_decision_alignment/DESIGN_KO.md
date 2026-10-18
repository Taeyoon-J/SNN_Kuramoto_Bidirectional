# SW0215: TRAIN 통계로 관계 loss와 merge 판정 기준을 일치

214 primary .720607은 원래 .788414보다 낮았다. Frozen geometry control의
.814008은 FG CI가0을 포함하고 ObjectIoU가 낮아 확정 개선으로 보지 않는다.
이후 점검에서214 loss는 positive/negative soft mass를 각각 normalize하지만
추론은 raw logit>0을 사용했다는 기준 차이를 확인했다.

Base image/pair weight w, soft target t에 대해 P=sum(w*t), N=sum(w*(1-t))다.
이상적 weighted BCE 해는 원래 target probability p의 logit에 log(N/P)를
더한다. 따라서 log(N/P)를 raw logit에서 빼고 기존 strict>0 complete-link
판정을 적용한다. 214의 초기 기준은 원래 p=.5와 일치하지 않았다.
이상적 loss 해의 보정이며 유한 NN의 확률 calibration을 보장하지 않는다.

값은 원래 TRAIN32 pair table에서만 계산한다. DEV score를 이용한 threshold
sweep이나 재학습은 없다. Frozen214 primary/geometry control을 CPU로 먼저
bitwise replay해야 새 예측을 평가한다. 모든 보정 mask를 고정한 뒤 GT를 읽고
원래 core, uncorrected214, corrected215 세 지표와 CI를 비교한다.

214 실패 판정을 유지하는 별도 posthoc-motivated 비교다. Core forward0,
optimizer update0, native64/고정16/seed2. Foreground와 원래 component 내부는
보존한다. 전체320/세 seed .90, 새70k 또는 모든 gating 인과 증거가 아니다.
