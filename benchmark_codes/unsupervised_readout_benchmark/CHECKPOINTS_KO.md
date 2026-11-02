# SW0259 전체 checkpoint — seed0/1/2

이 폴더의 checkpoint는 **FG-ARI 평균0.821628**을 기록한 기본 RGB arm이다.
성능이 낮아진 R1 prior-release 모델과 구분한다. 평가 조건과 해석의 한계는
[전체 결과](RESULTS_KO.md)에 기록했다. Foreground IoU0.218430은 배경을 지정하지
않는 출력의 raw 수치이며, 의미상 전경 검출 성능으로 해석하지 않는다.

| seed | 파일 | 크기 | FG-ARI |
|---|---|---:|---:|
|0|[sw0259_seed0.pt](checkpoints/sw0259_seed0.pt)|895,887 bytes|0.859454|
|1|[sw0259_seed1.pt](checkpoints/sw0259_seed1.pt)|895,887 bytes|0.796401|
|2|[sw0259_seed2.pt](checkpoints/sw0259_seed2.pt)|895,887 bytes|0.809030|

각 파일은 원래16×16 core 가중치 전체,32×32 mapping에 필요한 설정·integration
가중치, encoder 가중치·정규화 통계, 학습된 head와 복원 decoder 가중치를 포함한다.
Core를32×32로 확장한 반복 텐서를 중복 저장하지 않아 작다. 서버 경로에서 추가
checkpoint를 읽지 않는다. **Head만 저장한 파일이 아니다.** Optimizer·학습 데이터는
포함하지 않는다. Git LFS나 외부 Drive 다운로드 없이 이 브랜치에 직접 포함한다.

`snapshot/python/`은 실제 SW0165 실행에서 사용한59개 필수 소스·계약 파일의
원본 스냅샷이다. 새로운 Python 프로세스에서 이 스냅샷만 불러와 세 seed 각각의
전체 이미지 예측과 고정 모듈 state가 원본과 정확히 같음을 확인했다.
검증할 때 서버의 기존 소스가 대신 import되지 않았는지도 확인했다.

무결성 검증:

```sh
python -B benchmark_codes/unsupervised_readout_benchmark/verify_portable.py
```

추론 예시(저장소 루트에서 실행, 입력은 정확히128×128 RGB 이미지):

```sh
pip install -r benchmark_codes/requirements.txt
python -B benchmark_codes/unsupervised_readout_benchmark/portable.py \
  --checkpoint benchmark_codes/unsupervised_readout_benchmark/checkpoints/sw0259_seed0.pt \
  --image example128.png --output labels32.npy --device cuda
```

출력은32×32 정수 instance labels다. 입력 크기를 암묵적으로 변경하지 않는다.
CPU는 `--device cpu`로 실행할 수 있으나 core rollout은 오래 걸릴 수 있다.
관련 `snn_kuramoto_bidirectional` 모듈을 먼저 불러오지 않은 새 프로세스에서 실행한다.
가중치는 `torch.load(..., weights_only=True)`로 읽는다.

소스·checkpoint SHA는 [manifest](checkpoints/manifest.json), 새 프로세스 재현 증거는
[VALIDATION](checkpoints/VALIDATION.json), 원본 소스 SHA는 [snapshot/SHA256](snapshot/SHA256.json)에 있다.
기존 감독학습252 checkpoint를 재사용한 모델이 아니며, 고정 SW0165 core·encoder에
새로 초기화한252 구조 head·decoder를 mask 없이 학습한 모델이다.
