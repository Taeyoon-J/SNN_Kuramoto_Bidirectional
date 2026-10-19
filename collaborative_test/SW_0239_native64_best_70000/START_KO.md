# 239번 실제 학습 시작 확인

seed0가 GPU 2에서 실제 학습을 시작했다. GPU batch gradient와 native64 TRAIN
전체 forward 검증 후, 저장된 checkpoint에서 실제 Adam step 32와 이력 길이 32가
일치함을 확인했다. worker와 supervisor도 살아 있었다. 근거는 START_EVIDENCE.json이다.

이는 70,000장 학습 완료나 새 성능 점수가 아니다. 전체 계획은 seed당 70,000장·
10 complete passes·43,750개 logical Adam update이며 seed0→1→2를 GPU 하나로 처리한다.

첫 구간에서 HDF5 무작위 읽기의 CPU 압축 해제 병목도 확인했다. 동일한 RGB·mask·
학습 순서를 보존하는 입력 캐시를 240번에서 별도로 준비한다. 이 확인 시점에는
239번의 source·loss·초기 모델·학습 작업을 변경하거나 중단하지 않았다.
