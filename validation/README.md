# 실제 SW0240 재현 검증

- Endpoint: SW0249의 240번 seed0, 동일 35,000장×10회 추가 학습 종료 checkpoint.
- 원본 partial boundary를 기존 frozen core와 결합하여 전체 모델·encoder·학습 통계·teacher를 portable bundle에 저장했다.
- Bundle 파일 SHA와 복원된 student/teacher의 전체 모델 SHA를 검증했다.
- 로컬 CPU에서 실제 student/teacher를 로드하여 유한한 supervised loss와 실제 sensory projection의 양수 gradient norm을 확인했다. Optimizer update는 0회다.
- 서버에서는 복사한 폴더를 `python -I`로 실행했다. `collaborative_test`를 import하지 않았고, 로드한 메인 패키지의 모든 모듈 경로가 새 폴더 안에 있는지 검사했다.
- 원본 고정 평가 이미지 1320/1321/1322와 원래 inference seed 193000/193001/193002를 사용했다. 정답 mask는 읽지 않았다.
- 세 이미지의 64×64 최종 object mask가 기존 보관된 최종 예측과 label ID까지 정확히 일치했다. 원본 예측 파일과 RGB의 SHA도 확인했다.
- 검증 GPU: TITAN RTX, PyTorch 2.13.0+cu130, CUDA 13.0. 다른 GPU/라이브러리 환경에서의 bitwise 일치를 보장하지 않는다.

이 검증은 240번 seed0의 3개 이미지에 대한 실행 재현이다. 전체 평가 320장이나 세 seed 전체를 다시 평가하지 않았다. 성능 수치는 원래 완료된 최종 비교 결과를 인용한다. 새로운 학습이나 모델 수정 실험은 진행하지 않았다.
