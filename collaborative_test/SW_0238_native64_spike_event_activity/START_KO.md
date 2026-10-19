# SW0238 실제 복원 검증 및 큐 등록

로컬7검사, Linux7검사, CLI2개를 통과했다. 세 실제 SW0232 checkpoint를
CPU에서 복원하고 원래235번 accepted 모델과의 binding을 검증한 뒤 seed0/1/2
독점 GPU 큐를 등록했다. 추가 optimizer update는0이다.

첫 관측에서는 세 supervisor가 실제 살아 있었으나 모두 빈 GPU를 기다리고 있었다.
따라서 이 기록은 큐 등록 증거이며 실제6개 forward 완료나 진단 결과가 아니다.
`START_EVIDENCE.json`에 배포 SHA와 실제 관측을 보존한다.

기존237번 full320 평가는 변경하지 않는다. 70,000장 작업은 별도 범위 확인을
기다리고 있으며 이 진단으로 대체하지 않는다. 결과가 나와도 특정 TRAIN6개 trace의
관측으로 제한하며 전체 데이터 성능이나 gate 제거의 인과 효과를 주장하지 않는다.
