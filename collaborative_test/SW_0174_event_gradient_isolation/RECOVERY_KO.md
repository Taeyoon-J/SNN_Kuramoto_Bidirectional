# SW0174 CUDA 파일 읽기 오류와 동일 실험 재개

첫 실행은 source baseline의 `edge_rho.clamp_min(0).prod(dim=0)`에서
`basic_filebuf::underflow error reading the file: iostream error`로 종료됐다.
성능 결과나 학습 업데이트가 만들어지기 전의 실행 오류이며, 과학적 실패로 분류하지 않는다.
기존 parent 3786424 / worker 3786474의 종료를 확인하고 첫 실행 파일을 보존했다.

별도 GPU 1 진단에서 실제 사용한 여덟 edge 개수의 reduction forward/backward를
기본 캐시와 캐시 비활성 환경의 새로운 프로세스로 각각 실행했다. 두 실행 모두 성공했고
출력과 gradient 해시가 전부 일치했다. 기존 캐시 파일의 크기와 SHA도 유지됐다.
따라서 최초 오류는 이 작은 재현에서 반복되지 않았으며, 캐시 손상을 입증하지 못했다.

공식 PyTorch는 `USE_PYTORCH_KERNEL_CACHE=0`으로 CUDA JIT 파일 캐시를
비활성화하는 방법을 제공한다.
[PyTorch CUDA 문서](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/cuda.md)

이 환경 설정으로 동일 SW0174 코드를 새 `event_gradient_audit_v2_20261015`
실행 경로에서 재개했다. source SHA, 모델, loss, 입력, 비교 scale은 바꾸지 않았다.
실제 GPU 1 worker 3787993의 `/proc`에서 캐시 비활성 환경을 확인했다.
이는 원인 확정이 아닌 실행 우회이며, 개선 점수로 해석하지 않는다.
증거는 `results_archive/cache_bypass_recovery_20261015/`에 기록한다.
