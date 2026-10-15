# SW0174 결과: event surrogate 역전파 분리만으로는 개선되지 않음

별도 캐시 비활성 실행이 정상 종료됐다. 첫 CUDA 파일 읽기 오류와 구분한 과학적 결과다.
원본 SW0173과 같은 TRAIN cohort 4장 및 transfer 4장을 사용했고,
새 학습 업데이트나 고정 320장 평가를 실행하지 않았다.

동일 RNG 비교에서 gamma, 실제 A, projection, Q, 여섯 trace와 두 mask 출력이
정확히 일치했다. 여덟 source baseline도 기존 보고서와 일치했다.
65,536회 raw event의 역전파만 분리됐으며 scope와 원본 모델 상태가 복원됐다.

사전 지정한 full1024 / scale 1 결과:

| TRAIN 진단 항목 | 변경 전 | event 역전파 분리 후 |
|---|---:|---:|
| cohort Q loss | 0.26514053 | 0.26498338 |
| transfer Q loss | 0.35634674 | 0.35643243 |
| transfer 최종 source_F FG-ARI | 0.60379075 | 0.60171174 |
| transfer native_QCC FG-ARI | 0.60038102 | 0.60197111 |

앞단 전체 gradient norm은 context 755,906 / graph 505,522 / core 89,837로
여전히 매우 크며, full/tail 방향 cosine도 -0.89887이다.
기존 SW0173의 full gradient 폭증과 방향 불일치가 event 역전파 분리 후에도 남았다.
따라서 event surrogate만을 해당 문제의 원인으로 단정할 근거가 없다.

transfer loss 감소와 최종 mask 보존을 동시에 충족하지 못해 등록된 확대 기준은 실패했다.
64-update 학습으로 확대하지 않는다. 작은 scale의 일부 개선을 선택해 성공으로 보고하지 않는다.
이 결과는 모든 학습률에서의 실패나 spike forward의 무용함을 입증하지 않는다.

서버 raw report SHA: `7583c48c6909e2acdcf141a369c3e6b9f65c7c2681c146fb1e1ed226945e72c3`.
원시 결과와 독립 재계산은 `results_archive/completed_audit_20261015/`에 보존한다.
고정 32×32 세 seed 최고 기록은 SW0165의 **0.8043 / 0.6100 / 0.6511**로 유지한다.
