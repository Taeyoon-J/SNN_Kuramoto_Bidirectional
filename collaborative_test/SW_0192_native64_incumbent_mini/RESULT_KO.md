# SW0192: native64 실제 학습 검증 완료

SW0186 seed0의 learned RGB/context encoder, graph, core, integration을 native64로
옮겨 TRAIN8 이미지에 두 B4/micro1 QCE proposal을 수행했다. 모두 같은-RNG 실제
full-forward loss 감소 기준을 통과했다.

| proposal | 같은 batch 변경 전 loss | 변경 후 loss | 전체 시간 |
|---|---:|---:|---:|
| 1 | .288873 | .284535 | 445.05초 |
| 2 | .235884 | .227845 | 444.50초 |

두 행은 서로 다른 TRAIN batch다. 행 간 loss 차이는 학습곡선이 아니다.
실제 emitted-spike gradient와 다섯 parameter family의 gradient/proposed delta가
모두 유한하고 양수였다. 실제 Adam step2, 원본 CNN 변경, 최종 모델 hash와
완료 boundary/history도 검증했다.

- 평균 B4 proposal: 444.77초 (약7.41분)
- 두 proposal 합계: 약14.83분, 초기화/큐 대기 제외
- peak CUDA 할당 메모리: 5,374,827,008 bytes (약5.01GiB); driver 전체 점유와 다름
- gradient forward/backward: batch당 약286.8초
- 같은 조건 TRAIN64/16-proposal 단순 예상: 약1.98시간/seed

시간 예상은 두 proposal 실측을 일정하게 외삽한 값이다. 큐 대기·평가·연산 변동은
포함하지 않는다. 앞선 단일 추론 기반 예상보다 구체적인 학습 실측이며, 연산
최적화 후에는 다시 측정해야 한다.

이 결과는 학습 작동/자원 검증이며 mask 성능이나 gating 인과성 증명이 아니다.
70k 학습은 예약하지 않았다. SW0193이 같은32 development 이미지에서 추가 학습
전후 mask 성능을 측정한다.

최종 모델 SHA256:
`2dd2b0d64602b85d4082bcaddcd21dfe8c60d34af1ea78a2ed29bf03c8ba4b6e`

완료 boundary SHA256:
`14d0386db64d3890163574ac2421b1e80d3100536ec25c9940434e0689327e82`

보고서와 검증 증빙은 results_archive/completed_probe_v1/에 보관했다.
