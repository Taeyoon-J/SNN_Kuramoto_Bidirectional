# SW0164 prospective — SourceT foreground-preserving induced partition

2026-10-14. **GO: 하나의 고정 GT-free source-only assignment 시험으로 과학적으로 일관된다.** 구현/서버/queue 변경은 하지 않는다. SW162의 FG/object-IoU 개선과 FG-IoU 하락은 유지하며 그 failed expansion gate를 소급 변경하지 않는다. SW163는 그대로 진행한다.

## 질문과 고정 조건

SW162는 instance separation을 개선했지만 foreground/background membership도 바꿨다. SW164는 SourceT의 foreground membership을 정확히 고정해 **같은 support에서 instance partition만 개선되는가**를 시험한다. SourceT의 missing/false foreground를 고칠 수 없는 제한된 후보이며 .87이나 모든 gate 필요성을 보장하지 않는다. SourceT의 merge와 split 오류가 모두 존재하므로 single bridge가 주원인이라고 전제하지 않는다.

seed1 SourceT의 untouched native32 encoder/graph/core에서 B1,T1024/settle512/live_tail0를 사용한다. 같은 actual component spikes의 production Q와 실제 `prepare_coupling` tuple[0] A 및 learned gain을 SW162 방식 그대로 얻는다. Same-input source six traces/Q/QCC parity와 archived SW143T full320 control-label exact replay를 유지한다. SW159/158 intervention이나 새 학습은 없다.

Native SourceT `_qcc_labels(trace)`의 integer mask를 S라 하고 `F={n:S_n>0}`를 사용한다. GT가 아니라 **frozen model prediction**이다. SourceT의 largest-background/min8 처리는 이 source stage에서 이미 적용됐다. Source FG의 원래 instance IDs는 이후 solver에 넘기지 않고 membership만 사용한다.

## 정확한 induced solver

1. SW162 `signed_pair_energy(Q,A,gain)`으로 **원래1024-node full energy**를 계산한다. N/gain/epsilon/symmetrization/graph weights는 동일하다. F 크기로 N을 바꾸거나 Q/A/graph를 renormalize하지 않는다. Foreground-only Kuramoto를 다시 실행하지 않는다.
2. F 안의 각 node를 singleton group으로 초기화한다. Eligible adjacency는 original32-grid4-neighbor edge의 양 endpoint가 F에 있는 경우만이다. Background를 건너는 edge,diagonal/wrap,nonadjacent fallback을 추가하지 않는다.
3. Eligible group pair의 gain은 두 그룹의 **모든 cross-node full-energy 합**이다. Largest strictly-positive gain을 merge하고 exact ties는 두 group의 sorted first-patch pair로 결정한다. Dense cross sums는 nonadjacent groups에도 유지하며 adjacency는 quotient union으로 갱신한다. Positive eligible gain이 없으면 종료한다.
4. SourceT의 서로 다른 FG labels였던 nodes도 인접하고 gain이 양수이면 merge 가능하다. 반대로 source FG support의 서로 다른4-connected components는 join하지 않는다. 이로 인한 fragmentation 가능성을 숨기지 않는다.

## 작은 component와 background: total deterministic policy

**모든 F node를 foreground로 유지한다.** Size1~7 groups도 각자 positive label을 받는다. Nearest-group attachment,small-group drop,min8 재적용,new-largest-group background removal을 하지 않는다. 고립 node는 singleton FG label,empty F는 all0이다. Final foreground groups는 first row-major patch index 순서로1..K label을 부여하고 F 밖은 그대로0이다. K/scene count/GT count cap은 없다.

SourceT background 선택을 먼저 수행하므로 reference의 largest-background 정책을 임의 GT background로 바꾸지 않는다. 다만 **foreground refinement에 같은 filter를 다시 적용하면 support preservation과 모순**이므로 SW162 `canonical_labels(min_size=8)`를 final candidate에 그대로 재사용하면 안 된다. 이것은 명시적인 새 readout의 conditional-refinement 정책이며 modal32 target/metrics를 바꾸는 것이 아니다.

필수 invariant는 image마다 `candidate_labels>0 == sourceT_labels>0`의 exact equality와 모든 FG group의 induced4-connectivity/complete coverage다. 기존 `evaluation.py::patch_foreground_iou`는 nonzero labels를 합쳐 계산하므로 **fixed320의 per-image foregroundIoU가 SourceT와 정확히 같아야 한다.** 다르면 과학적 tradeoff가 아니라 구현/lineage 오류다. FG-ARI와 matched-objectIoU는 바뀔 수 있다.

## 작은 preflight와 fixed320 판정

고정8 TRAIN source positions `[0,1,2,3,2048,2049,2050,2051]`,optimizer0,TRAIN GT도 읽지 않는다. Source/capture parity,finite original Q/A/gain,deterministic energy/merge,FG exact membership,empty/singleton/disconnected-small-component cases를 확인한다. Source support size/topological component count,raw final group-size distribution,1~7-patch mass,count/runtime/peak memory를 diagnostic으로 기록한다. 이 값이나 TRAIN masks로 coefficient/merge rule/group count를 선택하지 않는다.

CPU solver는 SW162와 같은 incremental gains/adjacency/versioned heap을 사용한다. 새로운 GPU feature training은 없다. SW162 per-image archives는 Q/A **hashes**만 저장하며 actual matrices를 저장하지 않은 경로이므로 labels만으로 energy를 복구했다고 주장하지 않는다. 필요한 sourceB1 rollout을 한 번 실행해 control와 candidate에 공유하고 가능하면 SW162의 corresponding Q/A hashes도 재확인한다. Exclusive owner-aware reservation/per-image durable ledger를 사용한다.

seed1 fixed320 IDs1320..1639/native128 RGB/direct32 masks. All candidate/sourceT/Slot predictions와 protocols를 먼저 고정·검증한 뒤 modal4px GT와 원래 세 metrics를 읽는다. Reserve90640..90959는 읽지 않는다. Reference replay와 membership invariants가 깨지면 GT scoring 전에 technical stop한다.

Prospective extension gate는 FG mean>SourceT와 shared-image paired95%CI lower>0,objectIoU mean>=SourceT,foregroundIoU per-image exact SourceT다. Bootstrap10000,seed164를 고정한다. 통과할 때만 identical seed0/2 replication을 검토한다. 실패하면 min-size/nearest-attachment/neighbourhood/weight/threshold sweep 없이 닫는다. Extra predicted groups는 count diagnostic이며 modal-visible GT count를 hidden scene count나 model capacity ceiling으로 해석하지 않는다.

이 시험은 source-only unsupervised readout이다. 성공해도 fresh native32 70k/data scaling/retrained necessity/all-gate causality 또는 all3 Slot strict win을 입증하지 않는다. Frozen SourceT foreground error의 상한을 가진 conditional candidate이며 최종 .87/three-seed/70k 목표는 계속 미완료다.
