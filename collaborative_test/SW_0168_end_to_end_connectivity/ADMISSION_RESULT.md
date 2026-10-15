# SW0168 actual-spike 연결 학습: 선행 검증 통과

2026-10-15. TRAIN8, 각 arm 두 disposable B8 update(microbatch4), 실제 native32 T1024/settle512/live64 경로를 사용했다. **v4 큐의 terminal 상태는 complete_admitted**이며, root가 별도로 현재 로컬 구현 fingerprint에 대해 공개 report validator를 실행해 통과했다. 보고서 SHA256: `20814ce5dbe4cdb6680cfa69a0c82631c62de0d2a78e923367245f96fb49890a`.

| 첫 update의 raw gradient L2 | Maximin 연결 loss | Local-edge 비교 loss |
|---|---:|---:|
| RGB encoder | .19404750 | .09900457 |
| Native graph | .05740032 | .03371599 |
| Native core | .02205521 | .02518801 |
| Integration a_d/a_m/b 전체 | .000003698 | .000001714 |
| 실제 live-tail component spike gradient 평균 | .000648819 | .000381971 |

두 arm의 두 update 모두 encoder·graph·native core의 gradient와 실제 parameter delta가 양수였다. Native Q와 실제 component spike까지 gradient가 전달됐다. Integration credit은 작으며, 이것만으로 모든 scalar나 각 physical gating의 인과적 기여를 입증하지 않는다. 전체 gradient clipping은 두 arm/두 update 모두 작동할 필요가 없었다(scale1).

Maximin TRAIN loss는 .384801603→.383569217, local loss는 .259767124→.259165991이었다. 서로 다른 loss 크기를 성능 점수처럼 비교하지 않는다. 재사용한8장의 두 update 결과이므로 데이터 scaling이나 held-out FG-ARI 향상 증거도 아니다.

별도 frozen source B4 reference와 각 fresh arm의 초기 gamma·six-trace·Q·actual-A가 정확히 일치했다. Cache의 B4 component-spike/A도 일치했고, production native-B1 QCC와 SW164 zero-residual partition/merge-log의 live source replay8장도 각 arm 초기값과 정확히 일치했다. B4 cached-Q와 B1 Q의 bitwise 동일성을 요구하거나 주장하지 않는다.

GPU1에서 peak allocated memory는2,773,300,736 bytes, 각 B8 update는 약10.0–10.3초였다. 본 학습은 별도로 B16/micro4/64 update/512장 두 pass이므로, 이 자원 측정과 학습량을 구분한다. 검증 전후 source/context cache 파일 receipt는 같았고 VAL data는 읽지 않았다.

최초 세 attempt의 실행 문제는 모두 보존했다. v1은 QCC NumPy 반환값 처리, v2는 정수 QCC-label/boolean-F 입력 경계, v3는 GPU 완료 후 같은 process에서 CUDA 전용 검사를 재호출한 문제였다. v3는 양쪽 update 함수를 정상 종료한 뒤 보고서 저장 전에 실패했다. v4는 같은 immutable preflight_v3 학습 함수를 사용하면서 GPU 증거를 먼저 저장하고, 별도 CUDA-hidden CPU interpreter에서 source를 재검증해 이 문제를 해결했다. 이들은 FG-ARI 실패 결과가 아니다.

다음은 새 학습에서 conditional fixed-source support와 standalone learned-support를 모두 fixed320에서 평가하는 것이다. 최고 모델은 계속 SW165다. Gate 기여·.87 평균·native32 fresh70k scaling은 미달이다.

증거: `results_archive/admission_completed_20261015/server_records.json`, `independent_validation.json`. 실제 생성 파일은 `collaborative_testing/generated/` 아래에 보관한다.
