# SW0213: center head의 TRAIN 비작동을 단일 예산 확장으로 검증

211/212 실패 이후의 별도 실험이다. 211은 실제 TRAIN32에서도 center
검출 fallback100%, GT center의 .3 도달률0이었다. 반면 offset 오차는
무변위4.2434→실제입력1.5185 patch로 줄었다. 이 진단에 근거해, sparse
center 검출기가 작동할 만큼 학습됐는지 한 번만 추가 확인한다.

동일211 head/Adam/RNG128을 복사해 총512까지384 update를 추가한다.
원래211 checkpoint와128-step 실패 판정은 보존한다. TRAIN32/고정16/seed2,
loss·LR3e-4·width32·threshold .3 및 .5·capacity10은 모두 그대로다.
Core/encoder/gates는 동결하고 기존268-channel cache를 사용한다.

Primary는512 원래전경 보존 readout이다. 학습전경 readout은 별도 secondary,
TRAIN-mean/moments-only 입력 controls도 등록한다. 모든 endpoint를 먼저
고정하고 GT를 읽으며 validation에서 예산/threshold를 고르지 않는다.
TRAIN center 검출과 실제 Adam512/원본 history128 보존도 검증한다.

주요 구현은 기존211 training library를 그대로 사용한다. 새 child boundary의
steps512는 명시적으로 등록하고, 별도 origin에 부모128 boundary SHA와
bindings를 남긴다. Library fingerprint는211 dependency identity이며213의
전체 science fingerprint는 별도 admission/예측 protocol에 고정한다.

이 한 번의 예산 확장으로도 validation grouping 개선이 없으면 예산을
1024/2048로 반복 확대하지 않는다. Core forward0인 classifier 수렴 실험이며
새70k/전체320/세seed .90 혹은 모든 gating의 인과 기여 증거가 아니다.
