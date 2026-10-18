# SW0227 시작: 같은 checkpoint의 전체320장 확인

로컬9개·Linux9개 테스트, 3개 CLI 검사를 통과했다. 실제 comparable Slot 세 seed의
예측·protocol·hash·IDs 및 기존210 reference를 GT 없이 검증했다. 세223 checkpoint를
실제 CPU에서 복원해 전체320장 RGB와 원래16장 및225의64장 prefix를 검증했다.
CUDA 초기화와 optimizer update는 CPU admission에서 모두0이다.

세 seed 큐를 등록했고 관측 당시 seed0/2 worker는 GPU1/3에서 실행 중이었다.
각64장 캐시가 검증·재사용됐고 mask와 frequency의 추가 bitwise replay를 통과한 뒤
나머지256장씩 예측한다. seed1은 독점 GPU를 기다린다. 다른 사용자의 GPU 작업은
공유하거나 종료하지 않는다. 226번 데이터 증가 비교는 별도로 유지한다.

이번에는 새로운 학습이 없다. 223/225에서 선택한 고정 모델을 전체320장·세seed에서
기존 모델과 같은 Slot reference와 비교한다. 재계산된 source 및 Slot 평균은 원래
210 reference와1e-12 안에 같아야 한다. 세 지표 모두 Slot 평균 엄격 초과와 FG≥.90을
별도 판정한다. 이번 시작 기록에는 성능 결과가 없고 전체320장 incumbent는 유지한다.

225의64장 평균 .8772를 전체320장 결과라고 취급하지 않는다. 기존16장 gate 실패,
반복 validation 선택 및 mask-supervised lineage도 유지한다. 추가70,000장 학습이나
모든 gating의 인과 기여를 이번 테스트로 주장하지 않는다.
