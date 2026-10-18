# 동료 연구 확인 — 2026-10-18

224번 배포 후에도 origin/patch_v2를 fetch했고 HEAD는
7a21632b93805f4f1da69a2e77a9ddd95c624d0a로 변동이 없었다.
이번 가설은 실제 encoder의 valid3x3/context crop126 출력과64 adaptive pooling의
cell 중심을 대조한 자체 기하 진단에서 나왔다. 동료의 미검증 제안이나 단일 seed
headline을 포팅한 결과로 취급하지 않는다. 세 기존 source의 가중치를 유지해
동일 RGB/GT/ID/readout에서 새 입력 adapter만 비교한다.
