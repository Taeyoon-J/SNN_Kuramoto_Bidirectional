# SW_0136_native32_transfer

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `27fc7c2c0bf1a4ff17e90bb7d995d3fc1ad6cba05af5328b8795967b703b8ac2`

````markdown
# SW0136: completed-model transfer to native32

This prospectively registered screen transfers both completed SW0134 seed1 actual-spike arms to genuine 32×32 inference. Learned encoder, graph/core weights, all twelve integration values, full binder including its persistent noise buffer, and decoder weights are strictly preserved. Spatial node parameters use the frozen SW0135 mapping; RGB is freshly encoded at native128 and pooled to1024 patches.

The primary mask is direct32 learned-slot assignment. Native32 QCC and an independently loaded source97 seed1 QCC reference are separate readouts. Every320-image prediction is fixed before GT scoring; the same4px modal contract and Slot32 checkpoint predictions are used. No additional training, native32 training benefit, seed-mean victory, gating usefulness, or data-scaling result is established by this one-seed screen.

Server verification passed12 CPU tests and actual strict restoration of both completed checkpoints, with CUDA hidden. All three Slot prediction/protocol inputs were validated before the GPU queue launch. Supervisor2675399 and workers2675421/2675422 were observed live on physical GPUs0/1. The queue uses shared exclusive leases and terminates only its own worker if a foreign owner enters.

The final frozen research files were deployed once; no training retry or data change was needed. Completion and score evidence will be archived when the actual queue terminates. Research publication remains subject to the pending GitHub publication approval; local commits are not claimed to be pushed.

````

</details>
