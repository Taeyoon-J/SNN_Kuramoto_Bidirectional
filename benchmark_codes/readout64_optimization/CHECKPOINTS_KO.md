# 고정32 core + context64 head checkpoint

최종 모델은 **context_rgb**, 동일64 평가 FG-ARI 평균 **0.806885**다.
[전체 결과와 한계](RESULTS_KO.md)를 함께 확인한다.

`checkpoints/sw0260_context_seed{0,1,2}.pt`는 각각 약182KB로, 새 head·fine CNN·decoder를
저장한다. 원래 고정 core·encoder는 동일 저장소의
`../unsupervised_readout_benchmark/checkpoints/sw0259_seed{0,1,2}.pt`를 사용한다.
**새 파일만 단독으로 사용하는 형식이 아니며 같은 seed의 부모 checkpoint가 필요하다.**
부모는 이미 모든 core·encoder 가중치를 포함하고 있어 추가 서버 다운로드가 필요 없다.

```sh
python -B benchmark_codes/readout64_optimization/verify_portable.py
python -B benchmark_codes/readout64_optimization/portable.py \
  --checkpoint benchmark_codes/readout64_optimization/checkpoints/sw0260_context_seed0.pt \
  --image example128.png --output labels64.npy --device cuda
```

입력은 정확히128×128 uint8 RGB, 출력은64×64 정수 그룹ID다. 배경을 지정하지 않으며
출력 그룹을 모두 물체로 세면 안 된다. 새 Python 프로세스에서 실행한다.
필요 패키지는 `benchmark_codes/requirements.txt`를 참고한다.
`--device cpu`도 가능하지만 core rollout은 오래 걸릴 수 있다.

세 seed 모두 기존 서버 경로의 소스를 import하지 않는 별도 프로세스에서 전체
이미지→예측을 재현했다. [SHA manifest](checkpoints/manifest.json)와
[재현 검증](checkpoints/VALIDATION.json)을 보존했다. 데이터셋·정답 mask·optimizer는
이 추론 checkpoint에 포함하지 않는다. 원래 optimizer와 모든 연구 checkpoint는 서버에 보존한다.
