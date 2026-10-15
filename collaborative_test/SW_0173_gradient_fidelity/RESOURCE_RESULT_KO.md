# SW0173 전체 1,024 step 역전파 자원 검증

원래 SourceT seed1과 zero-residual contextual encoder를 유지한 native32 B1에서 전체 1,024 step 역전파가 정상 완료됐다. GPU 1에서 full forward/backward는 6.056초, peak allocated memory는 0.970 GiB였다. 이 측정은 TRAIN 첫 이미지 한 장이며 optimizer update는 0회다.

동일 RNG의 tail64/full1024 forward에서 gamma, 실제 Kuramoto coupling, graph projection, 여섯 trace 및 spike Q가 모두 정확히 같았다. 원래 source state SHA도 실행 전후 동일했다. 따라서 이 조건에서는 forward를 바꾸지 않고 gradient horizon만 비교할 수 있으며, 전체 역전파가 메모리 때문에 불가능하다는 가정은 근거가 없다.

Full1024 Q-loss gradient L2는 contextual encoder 371.8823, graph 241.1211, native core 120.6849, integration 1.0711e-6이다. Nonzero gradient와 자원 통과는 실제 update가 task를 개선한다는 증명이 아니다. 다음은 등록된 cohort/transfer에서 tail64/full1024의 projected fresh-Adam 방향을 실제 full forward로 비교하는 감사다. 아직 그 directional fidelity나 새 FG-ARI는 측정되지 않았다. 최고 기록 SW165는 유지한다.

로컬·서버 CPU 테스트 각각 13개가 통과했다. GPU 실행 전 실제 source와 TRAIN cache를 이용한 CPU native1024/tail64 forward-backward도 통과했다. 해당 CPU Q loss 0.1508268과 GPU Q loss 0.1507485의 차이를 CPU/GPU exact parity로 주장하지 않는다. Exact parity는 동일 GPU의 두 gradient horizon 사이에서 검증했다.

원본 report SHA256: `83a721329114d5ed948ce5ab210e15b97d17867f14f8517afdf48be7959d7e52`. Worker 3780648과 supervisor 3780598은 종료된 것을 확인했다. 근거는 `results_archive/completed_resource_20261015/server_records.json` 및 `independent_verification.json`이다. VAL RGB/GT, reserve 데이터는 읽지 않았다.
