# SW_0244_native64_within_precision

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `3938f6ba073f4dd021463ddf1271c5cb090d1b013570c4f2074e6b71aa1a4b32`

````markdown
# SW0244: matched within-object frequency precision training

SW0237 confirmed that longer optimization improves full320 FG-ARI to0.872680, but the current teacher loss does not explicitly prioritize high within-object frequency precision. SW0242 measured overlapping source frequency distributions. SW0243's component-gate substitution improved foreground coverage but worsened grouping and did not remove event saturation; close that frozen-weight branch.

Do not repeat the proposed threshold-calibration family: prior159 already moved all-on events to0.68–0.74 but lowered fixed320 FG-ARI from0.7540 to0.6688. Change the training objective instead. This is distinct from failed228, which increased source-center preservation, and failed230, which preserved source-relative inter-object distances.

Two matched arms start from the exact accepted232 seed1 checkpoint confirmed by237, at512 prior updates. Freeze the entire core, including the learned4x8 sensory projection, to avoid a cheap projection-collapse route. Train the whole encoder with identical fresh Adam groups/LRs, TRAIN64, batch16,128 new updates and identical batch/RNG order. Control uses the exact original teacher objective. Candidate adds15 times class-balanced within-object effective-frequency variance. Thus within variance has coefficient16 while the original source-center penalty and gamma anchor remain at1. Teacher arrays must match accepted223 arrays bitwise. No coefficient sweep is registered.

Use eight predetermined, evenly spaced native64 validation offsets0,40,80,120,160,200,240,280 from the same fixed320 contract. This avoids relying on the easy first-four prefix used for243. Keep the shared-gate native dynamics, T1024/settle512 and production spike readout. Freeze both arms' complete predictions and validate the parent237/Slot artifacts before reading evaluation GT. The original parent masks are a no-new-update reference; the control arm isolates extra optimization from the changed objective.

Foreground ARI is the primary selection signal. The within16 hypothesis needs FG at least0.01 above both its matched control and the original parent, with both IoUs strictly above the scoped Slot reference, before separate three-seed confirmation. A better control-only arm may justify further convergence confirmation, but is not evidence for the new objective. No eight-image result establishes the full320 three-seed0.90 target, a new incumbent, gating necessity or large-data scaling. Existing mask supervision remains explicit. The running70k job is unchanged.

Checkpoint actual encoder/Adam/RNG/history atomically and validate frozen core, teacher, input statistics and complete stage order. Source512 updates are provenance; the new Adam counter must be128, not640. Run local/Linux loss-gradient, core-freezing, scope and real interrupted-Adam tests, then actual CPU restoration/gradient admissions before GPU queues. Two independent exclusive queues may use free GPUs0..3 and yield only their owned workers to foreign users.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `575437ce9af7f4a9d6974ac37f6bd68058e1f892004a92832b38fa5d112c5dc3`

````markdown
# SW0244 result

On eight predetermined native64 validation images, seed1, strengthening class-balanced within-object frequency precision improved FG-ARI substantially. These are mask-supervised screening results, not the full320 three-seed benchmark.

| Model | FG-ARI | Foreground IoU | Matched-object IoU |
|---|---:|---:|---:|
| Parent237 | .857103 | .462292 | .690716 |
| Matched extra-training control | .864838 | .465304 | .696699 |
| Within16 loss | **.918801** | **.516084** | .642951 |
| Comparable Slot seed1 | .806661 | .208314 | .221033 |

Candidate versus matched control: +.053963 / +.050780 / -.053748. The registered FG-led expansion gate passed, but object IoU regressed. All three candidate metrics exceed this scoped Slot reference. Do not claim every metric improved versus our parent.

Both actual final encoder/Adam checkpoints were restored on CPU. Adam counters/history contain128 new updates; all core/projection parameters remained unchanged. Independent rescoring of32 masks (four models times eight images) reproduced all means/deltas within1.11e-16. Both original objectives started from the same512-update seed1 checkpoint and used identical data/batches. TRAIN within-object variance fell from .00701223 to .00274537 in the candidate, versus .00654130 in the control; source-center fidelity and gamma anchoring worsened, so this is a precision/grouping tradeoff, not proof of a solved mechanism.

Proceed to SW0245 with unchanged objectives and three seeds on64 evenly spaced validation images. The full320 incumbent remains SW0237 (.872680 / .626110 / .676331). The full320 FG0.90 target and70k scaling remain unproven. No gate contribution claim follows from this loss screen.

````

</details>
