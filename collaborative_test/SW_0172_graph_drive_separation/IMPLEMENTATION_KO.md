# SW0172 구현 기준

Frozen SW0134 rollout의 동역학 loop는 그대로 호출한다. 실제 GammaToDrive nn.Module의 임시 forward pre-hook으로 첫 입력만 선택하며, 종료·예외 시 hook을 제거한다. Graph에는 context gamma를 전달한다. Base CNN은 한 번만 계산하고 두 gamma에 같은 원본 normalization/clip/native32 pooling을 적용한다.

SW165 solver가 쓰는 adjacency는 raw graph output이 아니라 실제 Kuramoto prepare_coupling의 A_lat이다. Frozen SW0162 capture helper로 이를 별도 포착하고 raw graph adjacency와 구분한다. Live normalized z는 같은 실제 graph의 단일 projection에서 포착한다.

독립 로컬 CPU 검사 6개가 통과했다. 이 중 실제 S2NetCore/PhaseStateIntegration 1024-node, 4-step/live2 구성에서 같은 gamma를 두 입력에 주었을 때 frozen rollout과 six traces/Q/z/actual coupling이 정확히 같았다. 이 짧은 검사는 pretrained SourceT의 full1024/settle512 검증을 대체하지 않는다. 아직 GPU 실험·최적화 결과는 없다.

Control은 완료171 artifact를 재사용하지 않고 fresh shared-gamma 조건을 다시 학습한다. Candidate도 원래 SourceT에서 시작하며 실패한171 가중치를 이어받지 않는다. 다음은 원래 source/shared/separated의 실제 pretrained 초기 parity와 routing credit을 검증하는 bounded GPU preflight다. 학습은 통과 후 두 조건 동일 8-update prefix로 시작한다.
