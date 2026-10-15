# SW0167 조건부 설계 — 실제 연결 partition의 구조학습

2026-10-15. **설계만 작성; 구현·cache 생성·GPU 실행 승인 문서가 아니다.** SW163의 네 arm 최종 검증 결과가 아직 없으므로 지금 실행하지 않는다. SW165/166 및 다른 sealed source는 변경하지 않는다. 이 실험은 **TRAIN mask supervised track**이며 unsupervised 성적표에 합치지 않는다.

## 선택 이유와 진입 조건

완료 기록 `SW_0165_foreground_partition_replication/results_archive/completed_all_seeds_20261015/server_records.json`에서 fixed three-seed means는 FG/foregroundIoU/objectIoU = .804298915/.609975448/.651148975이고 Slot = .781651972/.204360128/.203164986이다. SourceT 대비 FG +.021254909 및 objectIoU +.054951846의 재현 개선이다. 이 성공한 가변 K, 실제 spike Q/actual coupling A, 4-connected foreground partition을 유지한다.

`SW_0166_fixed_support_ari_diagnosis/results_archive/completed_diagnosis_20261015/server_records.json`의 GT-assisted 순차 merge/split 교정 이득은 각각 .05530949/.05555731이다. 한 종류만 지배한다고 판단하지 않는다. .91516572는 connectivity 등을 완화한 **인증 상한**이며 이 solver로 도달 가능한 점수나 .87 보장이 아니다. GT attachment .87261955도 predictor가 아니다. 이 진단의 VAL GT를 TRAIN target이나 seed/계수 선택에 재사용하지 않는다.

SW163 Q_A가 등록된 gate를 통과하면 먼저 그 고정 recipe의 seed0/2 복제를 검토하고 이 후보는 대기한다. SW163가 통과하지 못하면 이 후보 하나를 수행한다. 슬롯 수·binder·RGB loss를 다시 바꾸는 대신, SW165에서 이미 효과가 확인된 전역 그룹화의 **학습 목적과 inference를 일치**시키는 가설이다. SW155의 pair MSE나 SW161/163의 슬롯 마스크 loss 반복이 아니다.

## 모델과 그대로 두는 항목

Seed1 SourceT encoder/graph/core/12 integration scalars는 모두 frozen. 실제 emitted component spikes에서 production 순서로 계산한 signed component correlations, Q와 native prepare_coupling tuple[0] A만 사용한다. RGB/gamma/theta/GT count는 classifier 입력이 아니다. SourceT QCC의 F={label>0}도 frozen. F 밖은0, F 안은 singleton까지 모두 positive label로 보존한다. 따라서 foregroundIoU는 per-image SW165와 정확히 같아야 한다.

SW164 `foreground_induced_merge`의 4-neighbor adjacency, strict-positive all-cross-pair greedy merge, canonical ties와 가변 K를 유지한다. 새 min8/background 재선택/attachment는 없다. 구현은 sealed solver를 바꾸지 않고 새 pure energy-input solver leaf를 만들며, 기존 operator와 전체 merge log 및 labels의 zero-residual 일치를 증명한다.

원래 signed energy E0_ij = wA_ij logit(Qbar_ij), wA_ij = 1+1024(Aij+Aji)/(2 gain). Qbar의 clipping과 gain은 sealed SW162/164와 동일하다. 공유 pair MLP는 10→32 ReLU→1, 최종 weight/bias=0이다. 대칭 입력은 [component rho 4개, logit(Qbar), log1p(1024 Asym/gain), 두 node의 Q row-mean 평균/절대차, Q row-std 평균/절대차]이다. Row 통계는 full1024, diagonal 제외이며 그 뒤 F를 유도한다. r_ij는 두 방향 평균으로 대칭화하고 Etheta=E0+wA*r, diagonal0. source를 담는 마지막0 합산까지 초기 energy/merge log/labels의 exact parity가 필요하다. 두 학습 arm은 파라미터 shape·초기값·입력·inference solver가 같다.

## 학습 target과 비미분 solver의 명시적 범위

TRAIN modal32 mask만 사용한다. F 안의 각 GT ID별 induced4-connected components를 분리해 feasible partition G*를 만든다. 같은 object라도 F 때문에 끊긴 부분은 별도 target component다. **F 안의 GT-background도 버리지 않고** ID0의 connected components로 target을 만든다; 이들은 inference 때 여전히 positive labels이며 GT로 background에 옮기지 않는다. 이는 fixed support의 한계를 감추지 않는 projected target이고 전체 GT partition과 같다고 주장하지 않는다.

Unordered pairs i<j in F에서 Y*=same feasible component. Positive/negative pair 집합이 모두 있으면 각 집합 총 weight를1/2로, 한 집합만 있으면 그 집합 총 weight를1로 둔다. Empty/singleton F는 두 loss 모두0이고 별도로 집계한다. 임의의 feasible partition G에 대해 D(G,G*)=sum a_ij |Y_G-Y*|, score S_theta(G)=sum Etheta_ij Y_G / Z, Z=max(sum |E0_ij|,1); Z는 frozen source에서 계산해 detach한다.

**Candidate: structured hinge.** Loss-augmented energy Eaug=Etheta+Z*a*(1-2Y*)를 detached CPU greedy solver에 넣는다. 그 결과와 G* 두 개 중 S_theta(G)+D(G,G*)가 큰 것을 Ghat으로 고른다(동률 G*). Loss = S_theta(Ghat)+D(Ghat,G*)-S_theta(G*). Memberships는 detach하고 선택된 두 partition의 score만 live tensor로 재계산한다. 따라서 gradient는 sum (Y_Ghat-Y*)/Z에서 pair MLP로 흐른다. Greedy oracle는 exact maximizer가 아니므로 **근사 structured surrogate**이며 true structured-hinge upper bound나 ARI의 미분이라고 부르지 않는다. G* 포함은 loss>=0을 보장할 뿐 최적 oracle을 보장하지 않는다.

**Matched control: local margin learning.** 같은 target weights로 sum a_ij ReLU(1-(2Y*-1)*Etheta_ij/c), c=max(mean |E0_ij|,1) detached. 같은 pair MLP를 학습하지만 실제 전역 partition을 loss 계산에 쓰지 않는다. Candidate/control 모두 최종 inference는 동일 global greedy partition이다. 이 대조가 새 capacity 자체와 실제 partition-level credit의 효과를 구분한다. Loss 혼합/추가 lambda/온도 sweep은 없다.

## 저비용 cache와 preflight

SW163의 검증된 seed1 frozen SourceT 512 TRAIN actual float32 S/Q/A cache가 source checkpoint, T power2/contrast8, RGB/order/encoder/stats, source-state hashes까지 일치할 때만 read-only 재사용한다. Q는 캐시 actual S의 production positive-clamped product다. Rho 추가 계산은 실제 production centering/normalization 순서로 하며 Q 재구성이 일치해야 한다. 새로운 teacher rollout이나 GT-derived affinity는 없다.

B4 TRAIN cache를 B1과 bitwise 같다고 가정하지 않는다. 원래 cache의 firstB4 replay/A capture proof를 검증하고 B4를 TRAIN arithmetic으로 등록한다. VAL은 SW165와 같은 B1/T1024/settle512/live0이다. Cache 안의 8 TRAIN에서 zero residual의 원래 SW164 partition 정확 재현을 확인한다. 별도 native B1 검증은 conditional admission 때 필요한 최소 4 TRAIN만 사용하며 기존 source hash·Q/A 경로와 연결한다; B4=B1 주장은 하지 않는다. GT는 TRAIN target 생성에만 사용한다.

CPU tiny connected examples에서 feasible target/BG fragments/singleton, loss-augmentation 부호, chosen-partition score의 analytical gradient, nonnegative loss를 확인한다. 두 disposable logical updates에서 초기 forward parity, 첫 final-layer update, 두 번째 hidden-layer credit 및 실제 changed weights를 확인한다. 일부 이미지의 hinge0는 정상이다. 모든 eligible TRAIN image에서 candidate gradient가0이면 actionable signal이 없으므로 중지하고 계수 변경으로 통과시키지 않는다. 16 TRAIN source examples로 CPU solver 시간/메모리를 측정한 뒤 전체 비용을 공개한다. 새1024-step rollout cache 수집 없이 시작하는 head-only 실험이다.

## 고정 pilot와 판정

두 arm 모두 같은 512 TRAIN, 8 complete passes=256 B16 updates, fixed shuffle167001/head167001, Adam LR3e-4, global clip1, fresh optimizer. 최종256 checkpoint 하나만 평가하고 validation checkpoint 선택은 없다. GPU micro4로 energy/MLP를 계산하고 CPU oracle는 detached energy만 받는다. optimizer/RNG/순서 포함 logical-update checkpoint를 사용한다. 예측 입력과 provenance를 freeze한 뒤 두 arm/SW165/SourceT/Slot의 fixed320 GT를 연다. Scene K/GT count/VAL oracle는 inference에 없다. Reserve90640..90959는 읽지 않는다.

Seed1 candidate FG가 SW164 seed1 및 matched local-control보다 모두 높고 두 paired95% CI lower>0, objectIoU>=SW164, foregroundIoU per-image exact일 때만 seed0/2 복제를 검토한다. CI는 common320 image indices/10000 draws/RNG167이며 seed-population CI가 아니다. 통과 실패 시 fixed recipe를 닫고 margin/MLP width/loss mix/seed sweep을 하지 않는다. 통과해도 .87, unsupervised 승리, all-gate causality, fresh native32 70k 또는 data scaling 증명이 아니다. 실제 full70k 확장은 all3 재현과 별도 supervised-versus-unsupervised claim 검토 후 새 등록으로만 진행한다.

현재 출력물은 이 DESIGN 하나다. 이후 생성되는 cache/prediction/checkpoint는 `collaborative_testing/prepare_output.ps1`로 중앙 output junction을 먼저 준비하고 provenance archive로 연결한다. 이 문서는 현재 SW163 실행이나 sealed165/166 source를 바꾸지 않는다.
