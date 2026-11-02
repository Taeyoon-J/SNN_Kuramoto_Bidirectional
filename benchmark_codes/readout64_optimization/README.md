# SW0260: 259 loss/readout optimization, updated output64 scope

Latest verified candidate: **context_rgb**, fixed320 three-seed64-grid FG-ARI
**0.806885**, versus old259 lifted/scored on the same64 targets **0.802087**.
[Results, controls and limits](RESULTS_KO.md), [checkpoints and offline inference](CHECKPOINTS_KO.md).
Core dynamics remain32; only fine features/head/decoder/output operate64.
The original encoder/core stay frozen; the new spatial context CNN belongs to the readout.
No TRAIN masks in weight learning. Labelled validation is used for checkpoint/config selection.
This is a measured local improvement, not a claim that loss tuning is globally finished.

The remaining text describes the first screening round; later delta/objective/context
controls and actual optimizer histories are preserved under `results/`.

Latest explicit user steering supersedes the goal tool's original native32 wording:
**keep existing SW0165/SW0259 core and encoder fixed,32-grid core dynamics;
perform head, image reconstruction and output masks at64x64.** No native64 core
or independent4096-oscillator claim. No TRAIN masks in loss/backprop/inference.

Coarse144D activity is repeated spatially from32 to64. A NEW zero-initialized8→32
adapter injects true64-grid gamma from the same frozen encoder into the head
embedding, so child cells can differ; this is not a final nearest-neighbor resize.
Original259 head embedding/decoder initialize from the mask-free RGB best checkpoint.
Candidate regions remain the original32 proposals, expanded to64, with local64-cell
competition. This controlled architecture change and its initialization are recorded.

Loss experiments remove the direct embedding→RGB shortcut using a frozen content
encoder, calibrate candidate similarities, and compare RGB reconstruction, edge-aware
RGB total variation, binned-spike neighborhood consistency and proposal self-anchoring.
Original published259 and all SW0259 outputs remain unchanged.

Fast screen: TRAIN128x8, all three seeds, five predeclared arms, common IDs/order;
promising configurations can extend to TRAIN256x20 with optimizer/RNG provenance.
Core activity caches are reused; no recurrent training or extra core rollout needed.
Validation32 IDs1000–1031 is reserved for checkpoint/config selection by FG-ARI,
with masks opened only in no-gradient validation. **This is labelled validation,
not mask-supervised weight learning**, and must be disclosed. Fixed320 IDs1320–1639
is not used for tuning; only score frozen final predictions. Compute old259's
upsampled predictions against the SAME64-grid targets;0.821628(native32) is not a
direct same-contract comparator at64. All-positive rawFGIoU still lacks semanticBG.

Local/server status observer runs quietly, read-only30s polls, with PID identity checks.
Report experiment starts/results and material failures; no routine epoch messages.
