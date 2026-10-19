# SW0230: source-relative 물체 간 주파수 간격 유지

223/225번은 내부 정렬을 개선했지만 229번에서 일부 false merge 증가가 관찰됐다.
228번의 절대 중심 고정 강화는 내부 정렬까지 약하게 만들어 FG 개선에 실패했다.
이번에는 원래 계수1 teacher denoising/gamma anchor를 유지하고, TRAIN 물체별
주파수 중심 사이 거리가 source보다 줄어드는 경우에만 부드러운 손실을 추가한다.
절대 중심 이동·공통 평행이동·거리 확대에는 추가 패널티가 없고 임의 unit margin도 없다.

각 이미지에서 존재하는 모든 class(BG 포함)의 unordered pair를 같은 비중으로
평균한다. d는 4개 frequency component의 평균 제곱 거리이며 추가 항은
ReLU(d_source-d_current)^2 / max(d_source,float32 epsilon)다. 계수는1로 고정한다.
원래 teacher/gamma 항도 유지하므로 전체 loss가 공통 이동에 불변인 것은 아니다.

같은 초기 모델·TRAIN64·128 Adam updates·batch16·LR/clip·encoder와 실제
4x8 projection 공동학습을 유지한다. graph/나머지 core/gates/readout은 고정한다.
TRAIN supervision만 사용하며 DEV는 고정16장·세 seed·원래 spike mask/RNG다.
대조군은 실제223번 계수1 모델이다. FG>=+.01/paired CI 하한>0 및223번 초과를
FG 후보 기준으로, FIoU/OIoU 평균 비감소를 balanced 기준으로 유지한다.
실패하면 이 설정을 종료하며 계수/LR/epoch sweep은 하지 않는다. 성공해도
전체320장 확인 전에는 최고 기록이나 .90 달성으로 승격하지 않는다.
