# SW_0241_native64_event_readout_counterfactual

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `2b5c2b21f71f4ae398eb52d9203e4b806a54819d0b5d3e9ff520a2ff0a9343fe`

````markdown
# SW0241: event information at the production readout

SW0238 found positive-gate event occupancy of 99.3265–100% on six actual TRAIN traces. Check whether the small remaining binary-event differences influence the final masks, before designing a learned membrane intervention.

Use exactly the same accepted232 seed0/1/2 checkpoints, two TRAIN images per seed and inference seeds as SW0238. One unchanged native64 T1024 forward supplies each actual trajectory. Keep its gates and graph fixed. The counterfactual replaces only downstream component spikes with the observed gates, rebuilds mean spikes, recomputes component correlations/Q and runs the identical production support/partition readout. Compare masks including foreground support and object-ID-invariant partitions.

This is an event-information readout intervention, not removal of membrane recurrence or dendritic feedback during dynamics. It neither trains a model nor measures held-out performance or causal necessity of the full membrane layer. No ground-truth labels are read. Retain full actual component spikes, gates, adjacency and paired masks for later exact recomputation; no raw arrays are published. Validate source/checkpoint/row/tensor hashes and unchanged model state.

Run CPU unit tests and real CPU restore admissions before the existing exclusive-GPU queue can execute. Other jobs, including SW0240 single-GPU TRAIN70000 and SW0237 full320 evaluation, remain independent. Six cases alone must not be generalized to the full data distribution.

````

</details>

## LAYOUT_CONTROL.md

<details>
<summary>LAYOUT_CONTROL.md</summary>

Source SHA256: `b037eed4295a5f05d4002e70a961ed586e411e875aa1039ef5f5b0f592e65c86`

````markdown
# SW0241 posthoc matched-layout control

Keep the original registered241 code and results unchanged. Reuse all six retained spike/gate/adjacency arrays. Rebuild actual component-spike strides using the exact stack-time-first, permute and reshape operations from the registered native rollout; gates and adjacency retain their canonical contiguous storage.

Before interpreting a new substitution, the identical production readout must reproduce both the original actual masks and the original contiguous gate-only masks bitwise. Copy gate values into storage with the actual spike strides, rebuild mean spikes and component correlations/Q, and run the production readout again. When the actual post-settle spikes equal the gates, require exact Q and mask equality as a negative control.

No optimizer update, GT input or native trajectory forward is added. The same three accepted232 models, six241 TRAIN cases, software/hardware math flags and downstream readout apply. Each seed has a fresh CPU restore admission and runs under the existing exclusive-GPU lease. The requested70k training is independent. All new source files are auxiliary and outside the frozen original241 fingerprint; a separate fingerprint binds this control to the original experiment and the rollout storage implementation.

Retain paired masks, exact replay checks and actual/legacy/matched strides. Aggregate only after all three control protocols validate. This can support a statement about event information at readout for these six cases. It does not remove membrane recurrence, prove population-level necessity, produce held-out performance scores or replace the current237 full320 reference.

````

</details>

## LAYOUT_RESULT_KO.md

<details>
<summary>LAYOUT_RESULT_KO.md</summary>

Source SHA256: `e03ab5fee3085bff23a378b4407d17a6473ef45095e852ac18b5c25e2bad6d05`

````markdown
# Matched-layout control completed

All six original actual masks and all six original contiguous gate-only masks replayed bitwise. The gate-only substitution was then recomputed in the exact native component-spike strides. Both seed1 negative controls, whose actual post-settle spikes equal their gates, retained bitwise-identical Q and masks.

The matched-layout substitution produced the same masks as the original substitution in all six cases. Four cases retained their actual masks; the same two cases changed. Seed0/case0 changed three foreground-support patches and seed2/case1 merged two groups while retaining foreground support. Therefore layout did not explain the observed changes in these six cases. Four cases had changed Q after event removal, but only two changed the final partition.

This is controlled evidence that the emitted event information can affect downstream grouping on these sampled TRAIN inputs. It does not show that either changed mask is more accurate, prove the necessity of membrane recurrence, or establish contributions for all gating layers. No GT, new trajectory forward or optimizer update was used. Full membrane/dendritic recurrence remained unchanged upstream.

The original241 experiment and limitation record remain preserved. This auxiliary control resolves that layout concern for the six observed cases. Saved paired masks and source bindings were revalidated at collection. The current full320 model reference remains SW0237 at0.872680 /0.626110 /0.676331; the requested70k training continues independently.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `ead8fef7552b2664ad977af635b8617147eac01f8c9255fbcb51bdd7e877c8ef`

````markdown
# SW0241 event information at readout

Compare actual emitted spikes with observed-gate-only readout on the same six frozen TRAIN cases as SW0238. Both mask paths use the production readout; the intervention rebuilds mean spikes and component correlations/Q. Full actual spikes, gates, adjacency and paired masks are retained privately.

This tests downstream event information. The native forward and its membrane/dendritic recurrence are unchanged. No ground-truth labels, optimizer updates or new performance claims are involved. An unchanged mask would not establish that removing membrane dynamics preserves performance.

CPU tests cover input preservation, production component-gate geometry, mean/Q recomputation, real native Q sensitivity, label-permutation-safe comparisons and correct exclusive-GPU worker routing. Real CPU checkpoint restoration must pass for all three seeds before GPU queues start. Source and generated results stay separate.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `69ffab9bb9d48bad3624ad856f26b3870e5d38e9f7c8be32355555a6ef507d9a`

````markdown
# SW0241: six-case gate-only substitution, with a layout limitation

Four of six frozen TRAIN cases retained identical masks. Seed0/case0 changed three foreground-support patches while retaining five objects. Seed2/case1 retained foreground support but changed the partition from23 objects to22. There is no GT-based score here, so neither change establishes an improvement.

The CPU archive audit re-read the actual full spike/gate tensors, exactly reproduced the prior238 tail-event statistics in all six cases, and recomputed comparisons from the saved paired masks. All source/checkpoint/row bindings and hashes passed. This audit does not re-execute the readout.

A post-completion code check found that the gate-only replacement preserved values but not native component-spike strides. Native rollout stacks time-first and returns a noncontiguous spike tensor; cloning the contiguous gate yields a different layout. Correlations are mathematically equivalent under layout changes, but floating-point reductions near a hard partition threshold can differ. The two observed mask differences therefore cannot yet be attributed exclusively to binary-event information. The all-one-event seed1 controls retained identical masks, which does not rule out numerical sensitivity in the other cases.

The required next control is a readout replay using retained arrays, with actual native spike strides restored and the gate-only signal copied into the same layout. First reproduce the original actual and original gate-only masks, then compare the matched-layout substitution. This needs no new training or native trajectory rollout. Preserve this original experiment and its limitation rather than changing its registered source after completion.

This is not a full membrane recurrence ablation, held-out performance result, proof of causal necessity or new best model. SW0237 remains the audited full320 reference. The ongoing70k training is unchanged.

Follow-up: the matched-layout control completed successfully. All original masks replayed bitwise and no substitution mask changed due to layout. The same two event-removal differences remained. See `LAYOUT_RESULT_KO.md` and `LAYOUT_RESULT.json`; attribution remains limited to downstream readout on these six cases.

````

</details>
