# SW0200 native64 settled-window time credit

가설: production 통계는 T1024 중 안정화 이후512시점의 spike를 쓰지만, 현재
직접 시간 역전파는 마지막64시점에만 걸린다. graph/drive는 setup에서 gradient를
유지하되 앞960시점 recurrent state는 detach된다. 이 truncated BPTT 편향이
추가 데이터 학습의 유용성을 제한할 가능성을 시험한다. 원인 확정은 아니다.

변경은 live-tail64 ->512 하나다. QCE temperature.1, 원래 full1024 forward,
settle512, 모든 gating/graph/CNN 및 생산 readout은 유지한다. 첫512시점의
recurrent history detach는 여전히 남는다. 전체1024 BPTT라고 부르지 않는다.
원본 geodesic 사용;196 가속 버전은 도입하지 않는다.

TRAIN8 source192 실제 model/Adam/RNG boundary2를 복원하고 indices8..15에서
B4 두 proposal을 추가한다. 199 control과 같은 시작 상태, 새 TRAIN8, LR5개,
clip1, same-RNG before/gradient/after accept/restore 정책이다. 각 proposal의
실제 S/five-family gradient를 기록한다. 기존 loss 비교군199 control은 이미
학습과16 mask 생성을 완료했고 두 proposal 모두 거절되어 warm model과 mask가
그대로다. 이 사실을 숨기지 않으며 양쪽 control/warm 대비로 평가한다.

GPU 사전검사: TRAIN index8 한 장에서 no-grad/live512 loss parity 및 실제 S/
모든 family gradient를 확인한다. optimizer step은 없고 model/Adam/RNG를 정확히
복원한다. allocator 최대비율.8을 적용하며 메모리 부족이면 이 후보는 reject하고
학습하지 않는다. 이 B1 gradient 평가는 추가 read-only TRAIN 노출로 별도 기록한다.
총16 unique/logical-proposal exposure는 inherited8 + 새8이며, 모든 forward 또는
gradient 계산 횟수라는 뜻이 아니다. 사전검사는 새 unique image를 추가하지 않는다.

고정 development offsets0,2,...30의16장, RNG193000+offset, 동일 GT/production
mask. 후보/비교군 protocol을 고정한 뒤 GT를 열고 세 지표와 paired bootstrap10000
(seed199;199과 같은 draws) 기록. 확장 gate: 후보-control FG>=.01 및 CI lower>0,
후보-warm FG mean>=0, 양쪽 objectIoU mean>=0, 새 accepted update>=1, 실제 model과
CNN의 warm 대비 committed change. 검증된 후보에만 추가 seed/full320를 진행한다.
이것은 seed0 supervised mini이며 최종 native64 three-seed .90 달성이 아니다.

199 control은194의 같은 QCE 첫4-proposal historical observation과 다르게
proposal4도 거절됐다. 초기 model/Adam tensor digest는 같고 측정된 family gradient
norm은 거의 같지만 역사적 optimizer trajectory의 완전 재현은 입증되지 않았다.
반면 고정 warm weights의16 mask는 과거 warm mask와 모두 정확히 일치했다.
본 비교에서 historical194 model을 control로 바꿔 끼우지 않는다. 동일 protocol의
199 실제 control을 그대로 사용한다. 수치 민감성 원인은 아직 확정하지 않는다.
