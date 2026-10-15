# SW0169 실제 GPU admission 통과

2026-10-15. 별도 v3 parent3713598/worker3713642(GPU1)가 `complete_admitted`로 정상 종료했다. 보고서 SHA256은 `91662b392f6a7b927cdc1da2bde411520107665df404329802d13dab9fc39b30`이다. Root가 현재 실제 source/cache/코드로 queue/public validator를 CPU에서 다시 실행하고, 아래 물리·gradient·업데이트 증거까지 독립 검증했다. CUDA는 이 재검증에서 초기화하지 않았다.

고정 TRAIN8·B4·전체1024/settle512/tail64 범위에서 원본 source와 opt-in source-mode의 six traces/Q/A가 정확히 같았다. 모든8행의 저장 S/A 및 사전에 정한 historical-layout cache Q도 정확히 재현했다. Source는 변하지 않았다. Active-current RMS float32 r=[2.812951565,2.679559708,2.744066715,2.076063871]를 GT 없이 고정한 후에만 별도 CPU interpreter에서 TRAIN targets를 읽었다. 새로운 r·threshold·time constant 탐색은 하지 않았다.

원본 source의 이 표본에서 E가 항상1이었던 상태와 달리, 새 초기 모델의 **g>0 조건부 raw E-on 비율은 image/component에 따라 .0399971.. .0731432**였다. 시간 전환은 g가 연속으로 활성인 pair에 대해서만 집계했다. 각 component의 전체8×1024 node-units 중 한 번 이상 이러한 E 전환이 발생한 비율은 다음과 같다. Gate-off에서 S가0인 것은 전환 증거로 세지 않았다.

| Component | 연속 active-gate에서 E가 전환한 node-unit 비율 |
|---|---:|
| 0 | 1.000000000 |
| 1 | .998535156 |
| 2 | .984863281 |
| 3 | .852539063 |

Live/control의 초기 six traces/events/Q hash는8행 모두 정확히 같다. 같은 maximin loss를 사용하고 직접 membrane-history credit만 다르다. 각 arm에서 실제 spike-derived Q를 사용한 disposable B8 update2회가 완료됐고 모든 microbatch의 실제 dL/dS norm이 양수였다(최솟값11.4435551). Encoder·graph·native core·기존 integration family가 모두 실제로 업데이트됐으며, 정확한 `kuramoto`, `dendric_layer`, `membrane_layer` 이름별 gradient와 변화량도 기록했다. tau_m와 b도 두 update 모두 바뀌었다.

실제 마지막 step의 값을 고정해 재생한 **local carry partial**에서 live의 previous V/E 감도 norm은 .0057430824/.0024383642, detached control은0/0이었다. 이것은 직접 carry 경로의 국소 미분 검증이며, control의 emitted-S→dendrite feedback까지 끊겼다거나 전체 시간 경로의 total gradient가0이라는 뜻이 아니다. 이 진단은 원래 학습 rollout의 gradient를 바꾸지 않는다.

두 update와 이후 TRAIN 진단에 걸린 시간은 live27.8722초/control27.6628초, peak allocated GPU memory는 약3.03GB였다. 이 시간에는 source-calibration/초기 parity 준비가 포함되지 않으며 B16 full-training update의 직접 시간 측정도 아니다.

**이는 선택적 발화와 학습 가능성의 admission 성공이다. FG-ARI/IoU 개선이나 gate의 물체 구분 기여는 아직 입증하지 않았다.** VAL은 읽지 않았고,512장·두 pass·64 logical B16 updates의 본 pilot과 fixed320 두 출력 평가가 다음 단계다. Fresh original source/tau/b와 동일 r로 시작하고 disposable update weights/Adam을 이어받지 않는다. SW165 incumbent(.804298915/.609975448/.651148975), 세 seed/.87/70k scaling 요구는 그대로다. 초기 두 구현 실패는 ADMISSION_START.md와 각 원본 archive에 보존했다.

증거: `results_archive/admission_completed_20261015/server_records.json`, `independent_verification.json`. 로컬·서버 CPU30 tests와 실제 queue CPU 진입 검증도 해당 packet에 있다. 생성물의 실제 위치는 중앙 `collaborative_testing/generated/`이다.
