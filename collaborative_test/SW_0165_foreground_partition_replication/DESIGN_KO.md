# SW0165 prospective — SW164 frozen-recipe seed0/2 replication

2026-10-14. **Conditional GO.** Root의 SW164 seed1 independent `validate_score` 완료 receipt와 frozen prediction/protocol hashes가 먼저 확인되어야 한다. 현재 보고된 seed1 candidate .770932/.614993/.630523 vs sourceT .755273/.614993/.579514의 긍정 신호를 복제하는 실험이다. 이 문서는 설계이며 서버/구현/queue를 변경하지 않는다. Sealed SW164 및 SW163는 불변이다.

## 변경하지 않는 과학적 recipe

동일 seed의 SW0097 source를 native16→32 strict spatial mapping한 SourceT,unchanged encoder/statistics/patcher,SW143 T power2/contrast8,actual emitted-spike Q,native prepare_coupling tuple[0] A,실제 per-source learned graph gain을 사용한다. B1,T1024,settle512,live_tail0,optimizer0이다. Threshold/emitter/graph weights나 energy coefficient를 바꾸지 않는다.

SW164 `foreground_signed.foreground_induced_merge`를 **그대로** 호출한다. SourceT labels>0의 induced4-neighbor support에서 singleton 초기화,unchanged full1024 signed energy,all-cross-pair gain,strictly-positive maximum merge,canonical first-patch ties를 사용한다. 모든 source foreground를 size1까지 유지한다. 새 min8 filter,largest-FG removal,nearest-group attachment/nonadjacency fallback,count cap을 추가하지 않는다. SourceT 단계의 native min8/largest-background policy는 그대로다.

SourceT와 candidate foreground membership의 exact equality,complete foreground coverage,각 final group의 induced4-connectivity는 모든 seed/image에서 필수다. ForegroundIoU는 per-image SourceT와 정확히 같아야 한다. 이 조건이 깨지면 새로운 tradeoff라고 해석하지 않고 technical failure로 멈춘다.

## Seed별 source와 proof

새 leaf runner에서 explicit seed parameter `(0,2)`를 처리한다. `SW130.run.source_contract(seed)`와 `SW135.foundation.load_native32_foundation(seed,device)`가 실제 generic API다. SW164/SW162의 public prepare/predict functions는 seed1/global output path에 묶여 있으므로 `module.SEED` monkeypatch나 문자열 치환으로 전체 pipeline을 재사용하지 않는다. 공유 pure solver/capture/trace-check 함수는 코드 fingerprint를 고정하여 재사용할 수 있다.

각 seed의 own source contract 순서에서 positions `[0,1,2,3,2048,2049,2050,2051]`를 고른다. Seed1의 image IDs/pool rows/model state/gain을 복사하지 않는다. TRAIN masks 없이 same-RNG untouched/capture eight-source parity와 source state unchanged,actual Q/A validity 및 empty/singleton/disconnected support invariants를 확인한다. Archived SW143 T seed별 native32 proof/reference를 검증하고 full320의 fresh QCC labels가 해당 T reference와 전부 일치해야 한다. 불일치는 technical stop이며 candidate를 조정하지 않는다.

Native source forward당 prepare_coupling에서 반환한 **동일 tuple의 첫 항**만 A로 캡처한다. Capture는 original tuple을 수정하지 않고 instance callable을 finally 복원한다. Per-image Q/A/gain/source state hashes를 candidate/control row ledger에 묶는다. Source gain을8이나 seed1 값으로 hardcode하지 않는다. 같은 forward를 두 readout이 공유한다.

## Fixed320/Slot과 실행 범위

Seed0와2 모두 IDs1320..1639의 동일 native128 RGB에서 직접32 labels를 만든다. 각 seed의 candidate/control/Slot archive와 prediction provenance를 고정·검증한 뒤 GT를 연다. SourceT own reference와 Slot own70k epoch10 seed가 섞이지 않아야 한다. Native128 Slot hard labels는 등록된4px modal converter만 사용하고 saved perimeter-BG policy를 유지한다. GT object count/allowed-label oracle/scene metadata를 inference에 쓰지 않는다. Reserve90640..90959는 읽지 않는다.

CPU source/reference admission 후 각 seed의 one eight-image TRAIN preflight,then one fixed320 B1 source pass가 범위다. Exclusive owner-aware leases/per-image transactional resume를 사용한다. Foreign GPU job은 공유·종료하지 않는다. 한 seed의 과학적 나쁜 결과가 다른 seed 보고를 취소하지 않는다. Technical failure는 해당 branch와 dependent scoring만 막고 partial artifacts를 보존한다. 완료 seed1은 검증된 결과를 재사용하며 수정/재학습하지 않는다.

## 보고와 판정

Seed0/2 결과는 좋고 나쁨과 관계없이 모두 보고한다. Discovery seed1과 prospective replication seed0/2를 구분한다. 각 seed 및 all3의 FG/foregroundIoU/objectIoU와 SourceT·동일 seed Slot 대비 차이를 기록한다. Shared320 sampled image indices를 모든 seeds/arms에 함께 적용한10000 paired bootstrap,RNG165를 사용한다. CI는 fixed three seeds에 조건부이며 seed-population generalization proof가 아니다. Seed0/2-only contrast도 별도로 보고한다.

고정 recipe의 replicated improvement는 all3 FG mean>SourceT,paired95%CI lower>0,적어도2/3 seed FG gains,objectIoU mean>=SourceT,foregroundIoU per-image exact SourceT로 판정한다. Ultimate board는 all3 means가 comparable Slot의 **세 지표 모두 strict superiority**인지와 FG mean>=.87을 별도로 판정한다. Discovery selection과 반복 사용된 validation의 한계를 밝힌다. Gate 미통과 시 threshold/weight/gain/min-size/seed selection sweep 없이 닫는다.

성공해도 source16-trained70k lineage의 native32 **inference/readout** 결과다. Fresh native32 70k training,positive unique-data scaling,all-gate causality,retrained layer necessity를 입증하거나 해당 jobs를 자동 승인하는 문서가 아니다.
