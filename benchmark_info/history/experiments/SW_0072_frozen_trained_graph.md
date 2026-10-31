# SW_0072_frozen_trained_graph

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `3e8bcce1a47632fa8688765d0fc327e0671c8569771bcf777c812443d9919453`

````markdown
# SW0072 - frozen trained seed0 graph

SW0065 shows that replacing only seed2's graph generator with the trained
seed0 graph rescues the 32-image result from `.3895/.2003/.1822` to
`.7036/.3081/.5376`. SW0066 fixed only its random initialization; training was
still free to move weak seeds back to different graph solutions.

SW0072 makes the causal intervention direct. Seeds 1 and 2 retain their own
non-graph initialization and data order, but load the completed SW0055 seed0
graph generator and keep it frozen for all ten epochs. Data, loss, core LR,
epoch count, and evaluation contract remain unchanged. The runner asserts that
every graph tensor is bitwise equal to the source after training.

## Result

The intervention passes both stages. On the 32-image pilot, seed1 improves
over SW0066 by `+.077105/+.082896/+.048197`, and seed2 improves by
`+.302338/+.075930/+.323319`. Every frozen graph tensor remains bitwise equal
to the trained seed0 source.

On the full fixed long contract, seed0/1/2 score
`.817116/.730071/.643890`, `.754097/.377045/.570852`, and
`.773010/.326999/.635497`. Their mean is **`.781407/.478038/.616746`**, a new
all-metric lead over SW0070 by **`+.103960/+.042599/+.136868`**. This is strong
causal evidence that a stable learned graph supports useful activation flow
through the trainable Kuramoto, dendritic, membrane, and spike stages. The
remaining gap is primarily foreground localization on seeds1/2 and FG-ARI
relative to the provisional official Slot checkpoint; the comparable trained
Slot three-seed baseline is still running.


````

</details>
