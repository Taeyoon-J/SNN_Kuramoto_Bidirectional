# SW0172 source admission 및 matched prefix 시작

최종 v4는 실제 GPU1에서 full1024/settle512/live64를 사용해 통과했다. Original source/shared/separated의 gamma, actual normalized projection z, 실제 factorized coupling A_lat, six traces, Q, source QCC와 SW165 energy/merge-log/labels가 정확히 같았다. 원본 source 상태의 전후 SHA도 같았다.

Candidate는 context→direct drive 경로가 없고, context→실제 graph→실제 S/Q의 양의 유한 gradient를 확인했다. 이는 학습 경로의 존재를 검증하며 FG-ARI 개선이나 gate의 유용성을 입증하지 않는다. 최적화는0회, VAL RGB/GT를 읽지 않았다. CPU 단계에서 기존 TRAIN512 mask cache를 검증·소비한 범위를 숨기지 않는다.

초기 세 시도는 모델 실패 결과에서 제외한다. v1은 중첩 --worker CPU 준비 중 GPU 지정 정보가 지워져 CUDA 전에 종료됐다. v2는 source1024와 후보의 기본64 steps를 잘못 비교했고 settle512 이후가 비어 Q=0이었다. 엄격 CPU source 로드로 기본값64를 확인했다. v3는 실제 theta[B,T,N,4]를 rank3로 잘못 검사했다. 이들은 구현·검증 오류이며 모델 성능이나 routing 가설의 실패 근거가 아니다. 이전 상태/로그/소스 SHA와 원문을 모두 보존했다.

수정 후 actual native CPU1024-node/1024-step 통합 검증과 실제 --worker 인수/CUDA 지정 조건의 CPU 검증을 통과했다. 초기 검증 관련 로컬·서버21개, 학습/큐까지 합친 로컬·서버30개 검사를 통과했다.

그 뒤 original SourceT에서 fresh shared_gamma 및 separated_gamma의 matched8-update prefix를 GPU2/1에서 시작했다. 실제 PID3767717/3767705의 NVML 점유와 각 durable boundary를 확인했다. 같은 TRAIN64 두 passes/B16/micro4/Adam/고정lambda를 사용하며, shared control에 완료171 weights를 재사용하지 않는다. 완료 결과는 별도로 기록하며 이 문서의 시작 기록을 현재 진행률이나 성능 결과로 해석하지 않는다. 최고 모델은SW165다.
