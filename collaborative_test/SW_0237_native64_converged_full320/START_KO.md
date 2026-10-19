# SW0237 전체320장 평가 시작

235번에서 검증한 실제 SW0232 누적512회 모델을 seed0/1/2 그대로 사용한다.
64장은 검증 후 재사용하고 나머지256장만 새로 추론한다. offset0의 mask와
frequency exact replay도 요구한다. 추가 학습은0회이며 기존 고정320장 계약과
comparable Slot Attention reference를 유지한다.

로컬15검사, 서버15검사와 CLI3개를 통과했다. 서버의 실제 CPU admission3개에서
checkpoint SHA와 source binding, CUDA 미초기화, 추가 optimizer update0을 확인했다.
세 독점 GPU 큐가 등록됐으며 관측 시 seed0/1은 GPU1/2에서 실제 worker가 살아 있고,
seed2 supervisor는 빈 GPU를 기다리고 있었다. 이미 새 mask 생성도 시작됐다.
이는 시작 증거이며 전체320장 결과가 아니다.

최초 서버 사전 검사에서는 incumbent 결과의 로컬 보관 경로만 가정한 문제가
검출돼 GPU를 시작하지 않았다. 서버 원본/로컬 archive 경로를 모두 검증하도록
고치고 실제 CPU admission에서도 확인했다. 실패 소스/상태는 서버
`results_archive/source_revision_v1/`에 보존했으며 수정 후15검사를 다시 통과했다.

`START_EVIDENCE.json`에 배포10개 파일 SHA, 실제 CPU복원3개와 checkpoint 식별자,
프로세스 생존·GPU 시도 기록을 보존했다. 원본 RGB·GT·checkpoint는 포함하지 않는다.
