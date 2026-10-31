# SW0254 모델/benchmark 계열

## joint_readout/DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `1468ec98cb46c6a38dbcf354760dfc01e86612d6a8c4bdc1e9c9cded087f78a6`

````markdown
# SW0254: 최종 mask loss의 encoder 공동 학습

기존252와 같은 source240, native64, TRAIN256/VAL64, 20epoch(5,120 Adam updates),
동일 head 초기화·CE+Dice·학습 순서·고정320·seed0/1/2를 사용한다.
기존 checkpoint와 benchmark는 수정하지 않는다.

변경은 최종 mask loss의 실제 gradient를 head에서 encoder 전체와 sensory projection까지
전달하는 것이다. 실제 1,024step forward를 유지하며 마지막64step만 recurrent gradient를
보존한다(960step prefix는 detach). 전체1,024step BPTT라고 부르지 않는다.
매 이미지의 graph와 hard partition은 현재 encoder로 다시 계산하지만 그 이산 결정 및
graph 생성 경로는 stop-gradient이다. Gradient는 gamma→sensory drive→Kuramoto→
sinusoidal/dendritic/membrane/spike→144D activity→head 경로를 통과한다.
Graph/core/gating parameter는 고정이며, surrogate gradient의 실제 유한·비영점 여부를
CUDA admission test와 학습 중 점검한다. 별도 frequency loss/anchor는 추가하지 않는다.

학습 head 초기 residual_scale=0 때문에 첫 update의 encoder gradient는0일 수 있다.
두 번째 update 이후 각 encoder/projection group의 연결을 검증한다.
학습 전에 실제 CUDA forward 동등성, backward, parameter 변경과 frozen state 보존,
checkpoint 저장/복원, benchmark preflight를 통과해야 전체 큐를 시작한다.

CLEVR best checkpoint는 VAL64로만 선택하고 fixed320은 학습/선택에 쓰지 않는다.
새 데이터는 그 CLEVR checkpoint의 zero-shot TEST360을 먼저 보고한 뒤 같은 checkpoint로
나머지640장을 평가한다. 별도 dataset adaptation은 TRAIN256×20의 공동 mask 학습이며,
기존252의 별도6,000장 frequency adaptation과 학습 절차가 다름을 결과에 명시한다.
새 데이터 학습 checkpoint도 VAL64로만 선택한다. 원본 pixel truth와 보호된 평가 함수를 유지한다.

배경을 지정하거나 삭제하지 않는252의 출력 규칙을 유지하므로 raw foreground IoU는
semantic foreground 판별 점수가 아니다. FG-ARI와 matched Object IoU를 주로 해석한다.

````

</details>
