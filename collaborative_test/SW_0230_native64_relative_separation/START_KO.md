# SW0230 서버 검증 시작

228번 계수16 중심 유지 설정은 종료했다. 이번에는 223번의 원래 계수1 teacher
denoising에 source-relative 물체 간 중심 거리 감소 패널티를 추가한다.
중심을 제자리에 고정하는 방식과 달리 중심 이동은 허용하고, 원래보다 가까워진
class pair에만 추가 패널티를 적용한다. 임의 단위 거리 margin은 사용하지 않는다.

같은 초기 checkpoint·TRAIN64·128 Adam updates·batch16·학습률/clip과
encoder+실제4x8 core 입력 projection 학습을 유지한다. graph와 나머지
core/gates/readout은 고정한다. mask supervision을 사용한다. 고정16장·세 seed
대조는 실제223번 모델과 진행하고, 세 지표 및 사전 등록 후보 기준으로 판단한다.

로컬 테스트15개를 통과한 버전만 서버에 보냈다. 서버에서도 테스트15개와
train/evaluate/queue CLI 검사3개를 통과했고, 세 seed의 실제 데이터·모델·teacher와
projection gradient CPU 검사를 시작했다. 이 검사가 모두 통과해야 GPU 큐가
등록된다. 이 문서는 서버 검증 시작 시점의 기록이며 GPU 학습 완료·성능 결과는 아니다.

서버 launcher PID2759128, CPU 검사 PID2759175/2759176/2759177.
배포 파일별 SHA는 START_EVIDENCE.json에 기록한다. 외부 GPU 사용자가 있으면
기다리고, 학습 중 외부 사용자가 나타나면 자기 작업만 양보하는 기존 큐를 유지한다.
기존227번 전체320장 평가는 중단하거나 재시작하지 않는다.

동료의 최신5c88a6f 결과는 PEER_REVIEW_KO.md에 검토했다.
이 실험 결과가 확인되기 전까지 전체320장 최고 기록은 .848615/.592533/.672047이다.
