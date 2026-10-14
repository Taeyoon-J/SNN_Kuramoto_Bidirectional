# SW0162: sourceT graph-aware connected signed partition

2026-10-14 prospective design. 구현·학습·서버 실행은 하지 않는다. SW161 및 이전 source 파일은 불변이다. 이 시험은 GT-free **source-only assignment 구조**를 비교하며 optimizer updates=0이다.

## Source와 실제 A 캡처

seed1의 `SW0135/foundation.load_native32_foundation(1, device)`를 strict source97 mapping으로 로드하고 `SW0143/geodesic_operator.install_operator(graph,'T')`를 설치한다. 기존 feature statistics/encoder/native32 patcher를 그대로 사용한다. Feedback는 꺼져 있고 factorized coupling, static gamma, spike pulse 없음이어야 한다. 별도 threshold·emitter·encoder·signed-coupling intervention은 설치하지 않는다.

원래 SW143 prediction은 B1, T1024, settle512, live_tail0이며 `SW0136/evaluate._predict_one(...,primary=False)`의 SW134 late rollout과 `_qcc_labels`를 사용한다. 같은 setting과 RGB/order를 재현한다. Gamma는 실제 uint8 RGB의 source encoder에서 재생성하며 gamma16 cache/upsampled labels를 사용하지 않는다.

실제 native `GraphVectorKuramoto.prepare_coupling`은 `(A_lat, lag_cos, lag_sin)`을 반환한다. 첫 항은 `relu(.5*(rawA+rawA.T))+1e-6`이다. **이 호출의 반환 tuple[0]을 캡처한 값만 A로 사용**한다. Caller instance의 original bound prepare callable을 잠시 감싸서 original을 한 번 호출하고 반환 tuple[0]을 clone/detach한 뒤, 동일 tuple을 수정 없이 native rollout에 반환한다. finally에서 callable을 복원한다. Global monkeypatch, A 재계산, correlation을 A로 둔갑시키기, signed coupling 설치는 금지한다. Static source에서는 image당 prepare 호출이 정확히 한 번이어야 한다.

Q는 반환된 actual component spikes의 settled512에서 production `spike_synchrony_affinity(...,settle=512,affinity_mode='spike')`로 계산한다. Mean spikes를 네 component로 복제하지 않는다. Positive-clamped component product는 원본 그대로다. A/Q의 original float32 bytes/hash와 shape를 기록한다. Solver만 CPU float64 copy를 사용한다. Gain은 실제 source `exp(graph.log_coupling_gain)`이며 contrast8 또는 Kuramoto K1024를 gain 대신 쓰지 않는다.

## 고정 energy와 greedy rule

N=1024. Off-diagonal에만 아래를 계산한다.

`q_s=(Q.double()+Q.double().T)/2`

`q_bar=clamp(q_s,1e-6,1-1e-6)`

`w_ij=1+N*(A_ij+A_ji)/(2*gain)`

`e_ij=w_ij*log(q_bar_ij/(1-q_bar_ij))`

Diagonal e는0이다. Q>=.5의 기존 의미를 log-odds0으로 보존한다. Symmetrization/epsilon은 고정 numerical definition이며 threshold/weight search를 하지 않는다. A의 native1e-6 floor도 보존한다.

1. Row-major32 grid의1024 singleton groups에서 시작한다.
2. 두 그룹 사이에 original4-neighbor(horizontal/vertical,wrap 없음) grid edge가 하나라도 있을 때만 merge 후보다. 대각 neighbor/learned adjacency만으로 merge 허용 여부를 바꾸지 않는다.
3. Gain(G,H)=sum_{i in G,j in H} e_ij. **두 그룹의 모든 cross pairs를 합한다.** Grid edge만 합하면 다른 방법이 된다.
4. 가장 큰 strictly-positive gain을 가진 eligible pair를 merge한다. Exact equal gain이면 `(minpatch(G),minpatch(H))`의 sorted tuple이 lexicographically 작은 pair를 고른다. 새 group ID는 작은 minpatch다.
5. Group adjacency는 union으로 갱신하고 모든 remaining group과의 gain은 기존 두 gain의 합으로 갱신한다. Nonadjacent groups의 dense gain도 보존하여 나중에 adjacency가 생길 때 사용한다.
6. Positive eligible gain이 없으면 종료한다. K,merge count,foreground fraction을 강제하지 않는다. 이는 deterministic greedy heuristic이며 global multicut optimum이라고 부르지 않는다.

CPU dense energy/gain arrays와 adjacency sets,versioned max-heap으로 incremental 갱신한다. 모든 merge마다 전체 dense matrix를 다시 계산하거나 threshold/refinement sweep을 하지 않는다.

## 공통 mask/evaluation 규칙

Candidate group 중 size8 이상을 유지하고, 가장 큰 group을 background로 제거한다. 동률은 가장 작은 first patch index로 결정한다. Size8 미만과 선택된 background의 labels는0. 남은 groups는 first-patch 순서로 positive IDs를 부여한다. 유지되는 group이 없으면 all0이다. 이 경우를 오류로 숨기거나 다른 decoder로 바꾸지 않는다.

Control은 original `SW0136/evaluate._qcc_labels(trace)` 그대로다. Capture 전/후 native six traces와 Q가 exact equality여야 한다. Seed1 fixed320 control labels가 validated SW143 `T/seed1` archived labels와 모든 이미지에서 정확히 일치하지 않으면 technical failure로 중단하며 GT를 읽지 않는다.

Candidate는 actual spikes+실제 couplingA+고정 grid topology만 사용하는 새 readout이다. RGB/GT/γ/θ를 solver feature로 추가하지 않는다. IDs1320..1639,modal4px/native32 targets,original FG-ARI/foregroundIoU/objectIoU scorer 및 Slot native128 archive→modal4px contract는 변경하지 않는다. 모든 control/candidate/Slot predictions와 provenance를 고정·검증한 후에만 GT masks를 연다.

## 8 TRAIN preflight와 budget

Source immutable4096 order의 positions `[0,1,2,3,2048,2049,2050,2051]`를 사용한다. 각 이미지를 B1/T1024/live_tail0로 실행한다. TRAIN masks도 읽지 않는다. 동일 RNG state에서 untouched/captured rollout의 six-trace/Q/QCC exact parity,actual A finite/symmetric/positive shape `[1,1024,1024]`,finite positive gain,source parameter/buffer unchanged,solver determinism/partition coverage/connectivity를 검사한다.

Image별 solver/rollout seconds,peak CPU RSS/CUDA memory,merge count,raw group size distribution,size8 retained group counts,chosen BG size,discarded mass,candidate/control predictedFG coverage를 기록한다. Dense negative energy가 BG를 조각내고 object를 largest BG로 선택할 위험이 있다. **TRAIN GT 또는 group count/FG coverage 수치로 admission/parameter 선택을 하지 않는다.** 이 수치들은 diagnostic이다. CPU/GPU OOM/invalid finite values/parity 오류는 technical stop이다. 기술적으로 feasible하면 preflight의 모양을 보고 recipe를 바꾸지 않고 full320을 보고한다.

통과 후 seed1 B1 full320 한 rollout/image에서 동일Q/A의 두 readout을 저장한다. Exclusive owner-aware GPU lease를 사용하며 foreign process는 공유·종료하지 않는다. 행별 durable ledger와 current source/file fingerprints를 묶어 interrupted work를 안전하게 보존한다. Runtime limit을 발견하면 parameter를 바꿔 우회하지 않고 operational blocker로 기록한다.

## Frozen decision

Seed1 expansion gate: candidate-minus-sourceT mean FG>0 및10,000 paired-image bootstrap95%CI lower>0,foregroundIoU mean>=sourceT mean,objectIoU mean>=sourceT mean. Bootstrap RNG162,image indices를 두 arm에 공유한다. Endpoint 결과를 본 threshold/weight/K 변경은 없다. 실패하면 이 recipe를 닫는다. 다른 지표의 유용한 결과는 selection policy에 따라 별도 track으로 기록하되 seed expansion gate를 통과했다고 부르지 않는다.

통과할 때만 동일 source0/2 recipe를 replication 대상으로 검토한다. All3 means가 saved own70kSlot의 세 metrics를 strictly 넘는지와 FG .87 목표를 따로 보고한다. Seed1 win은 최종 성공이 아니다. 이 source-only inference test는 fresh native32 70k training,positive data scaling,retrained layer necessity 또는 모든 gate의 causal contribution을 입증하지 않는다. 새로운 training objective/70k jobs는 여기서 자동 등록하지 않는다.
