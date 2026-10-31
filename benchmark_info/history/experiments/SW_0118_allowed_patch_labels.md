# SW_0118_allowed_patch_labels

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `6af9364416fed8f88185309ef2d8e934eaecac5a81b18647d6b7e0a6d2e32bc0`

````markdown
# SW0118 allowed patch labels

SW0118 is a loss-free diagnostic applied after a frozen prediction is available. For each 16×16 predicted patch, it builds the set of instance IDs appearing in that patch's 8×8 GT pixels, including background. The original majority target still comes from `clevr_mask_patch(mask, 8)`, so modal ties continue to choose the smallest ID.

Predicted groups are matched one-to-one to non-background GT IDs by a rectangular Hungarian assignment. The integer objective counts how many patches in a predicted group allow each GT ID. Predicted groups are ordered by first row-major occurrence and GT IDs numerically; this gives stable deterministic tie handling without changing the objective with epsilon. A group with no positive-overlap match receives a unique negative sentinel that is never allowed.

At each patch, the candidate label is 0 for predicted background and the mapped GT ID for a non-background group. Acceptance records whether that candidate is present among the patch's pixel labels. If it is not allowed, the original modal GT label is retained; thus a predicted background on a pure-object patch does not erase that foreground target. The scorer asserts every resolved target is an allowed pixel label, then applies existing `patch_fg_ari` to the unchanged prediction and resolved labels. This is named **allowed-label oracle FG-ARI (one-to-one membership matching)**. It uses GT after prediction and is not exact ARI optimization or a standard model metric. It can show whether predicted groups align with any pixel-level label present in their footprints; it cannot replace or modify registered FG-ARI, IoU, or matched-object-IoU results.

The result includes acceptance rate/count, original-majority and resolved foreground counts, changed target count, and per-group mapping/overlap counts. Undefined scores for fewer than two resolved foreground patches are `null` with `score_valid=false`, keeping JSON output valid. Inputs are not mutated.

`run.py --output <new-directory>` validates all frozen prediction blobs, source/checkpoint/gamma hashes, IDs, and existing score reports before opening validation masks. It then rechecks the original metrics against the frozen SW0097 per-image reports and matched Slot summaries. The released Slot checkpoint is shown separately because its training data differs; it is not a matched three-seed baseline. Output is diagnostic only and cannot affect model selection. No model, training, thresholds, or promotion gates are changed.

````

</details>
