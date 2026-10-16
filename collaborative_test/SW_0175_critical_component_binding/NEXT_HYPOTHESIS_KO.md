# SW175 평가 후 조건부 다음 가설

이 문서는 다음 후보에 대한 설계 검토이며 새 실험의 실행 또는 성능 결과가 아니다. SW175 고정320 평가가 완료되기 전에는 해당 후보를 학습하지 않는다.

SW173에서는 전체1024-step gradient가 tail64와 크게 다른 방향이고 크기도 컸다. SW174에서 raw-event surrogate backward 경로만 제거해도 폭증과 음의 방향 정렬이 거의 남았다. 따라서 event surrogate 하나가 원인이라는 설명은 기각하며 같은 작은 loss 조정을 반복하지 않는다.

SW175 후보가 두 endpoint의 등록 조건을 통과하지 못하면, 다음 단일 후보는 **전체 settled frames에 local phase-parameter gradient를 주는 학습 정책**이다. 각 Kuramoto 전이의 theta_prev 입력만 detach하고 그 step이 내보낸 theta output은 live history에 둔다. 실제 코드에서 과거 history는 다음 Kuramoto 입력으로 재사용되지 않고 sinusoidal carrier/delayed gate에서 읽히므로, Kuramoto 전이 사이의 연속 Jacobian만 끊으면서 각 delayed gate에서 해당 local step의 parameter로 흐르는 gradient는 유지할 수 있다. D/M/event/reset의 장기 gradient까지 모두 제거한다는 주장은 하지 않는다.

Forward 값은 바꾸지 않는 approximate recurrent gradient 정책이다. six traces, actual Q/A 및 두 mask가 원본과 같은지 먼저 확인하고, 작은 graph 검사로 successive theta gradient 차단과 delayed gate의 local parameter gradient 보존을 증명해야 한다. 이후 실제 GPU에서 gradient 크기·방향과 첫 update의 자원 사용을 측정한다. 안정화 효과는 아직 입증되지 않았다.

loss는 기존 critical-component, source 및 512장×2 TRAIN 순서, B16/micro4, optimizer/LR/clip, 64 updates를 고정한다. 완료된 SW175 critical-component 모델은 해당 조건과 초기/RNG/boundary lineage가 모두 맞는 경우에만 tail64 대조군으로 재사용한다. 후보는 실패한 가중치가 아닌 동일한 fresh source에서 시작한다.

확대는 proxy loss 감소로 결정하지 않는다. 동일 고정320에서 conditional/standalone FG-ARI가 기존 source와 tail-control 대비 paired bootstrap 조건을 통과하고 두 IoU를 유지해야 한다. 실패하면 이 정책도 닫는다. 성공해도 모든 gating의 필요성, FG-ARI .87, native32 fresh70k 및 데이터 증가 효과는 별도의 검증 대상이다.
