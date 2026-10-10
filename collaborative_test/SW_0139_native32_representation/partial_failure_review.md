# SW0139 partial failure review

Sol6.1 design review, 2026-10-10. This document uses completed control/context records only. Cross-view training and SW0140 full320 causal endpoints remain pending; their outcomes are not inferred here. No running recipe, source file, queue or guard is changed.

## Established outcome and implementation scope

The fixed native32 seed1 source FG/foreground-IoU/object-IoU is `.754135/.606900/.582201`. Old-loss encoder/graph continuation gives `.726187/.592371/.557006`; context residual gives `.666608/.650807/.477581`. Control-minus-source FG is `−.027948`, paired CI `[−.039005,−.017387]`. Context-minus-control FG is `−.059579`, CI `[−.077038,−.042826]`; object-IoU also falls, while foreground-IoU increases. Close this context recipe: no other-seed expansion, dilation/LR refinements or 70k continuation.

Both completed histories have the same 4096 ordered images, 256 logicalB16/B4×4 updates and native1024/settle512/tail64 rollout. Actual `trainable_backbone` releases encoder and graph only; context adds its zero-output multiscale residual. Native drive/Kuramoto/dendrite/membrane and all 12 integration/threshold scalars remain frozen. Therefore inputs and learned coupling can drift substantially while downstream state parameters remain unchanged. These results do not establish a defective membrane learning rule or a useful event selector.

Evaluation regenerates gamma from each trained encoder plus its context module, then uses actual spikes and unchanged QCC. It is not scoring an auxiliary RGB head. Residual features are clamped to the registered range before pool32. The completed numerical/lineage checks establish a faithfully evaluated failed recipe, not a demonstrated implementation mistake.

## What explains the pattern, and what does not

The implemented objective is phase criterion plus 5×actual positive-clamped product-Q criterion. Each criterion applies bimodality6, density balance10 toward `.867`, collapse barrier1 and coherence1. The coherence term smooths each node's **mean synchrony with all other nodes** across the grid; it does not directly require correct object boundaries or within-instance relations. These global statistics allow many distinct partitions. Objective/instance mismatch is therefore plausible, but density alone is not identified as the cause, and no target should be chosen from GT.

Descriptively, first32/last32 old-loss means are `26.190→25.064` for control and `27.737→18.622` for context. The batches differ, so these are not paired loss improvements. Nonetheless endpoint degradation alongside strong progress through that objective provides no reason to enlarge the same context mechanism. Higher foreground IoU with worse FG-ARI/object IoU warns of partition errors; merge versus split versus background reassignment must come from Luna's pending CPU diagnosis, not this metric pattern alone.

Every update clips the unique parameter union at1. Median preclip norms are `5.486/7.602`, multipliers `.1823/.1315`. Context adds parameter capacity, a different LR and a gradient direction to that union, so its comparison combines representational change and optimization effects. These measurements neither locate family-specific credit nor prove clipping caused the failure. With Adam, those multipliers are not simply effective LR ratios. No clip/LR retry is justified by the partial results.

## Conditional next priority

First retain SW0137 as incumbent and finish cross-view and all six SW0140 interventions on all three seeds. No new training admission now. Do not duplicate the requested CPU partition diagnosis or inspect reserved IDs.

If cross-view preserves source FG and provides credible paired benefit, and SW0140 shows native downstream history/events provide no mask benefit, the next mechanism question is the already documented **task event-credit versus terminal event-detach pair**, releasing only native membrane retention and component thresholds in addition to encoder/graph. Keep actual1024 spikes, native recurrence and QCC; require literal E×g forward parity, finite selector credit and actual event-decision changes on fixed TRAIN audits. A gradient or lower event rate cannot establish useful gating.

If cross-view also degrades source, do not promote the same objective to more capacity or release downstream selectors automatically. The next broad objective question should instead preserve source binding while learning view stability: a fresh prospective comparison of the completed old+C continuation against **C plus detached frozen-source actual-Q preservation**, replacing the global old-prior optimization as a single declared objective change. This preserves a numerical incumbent relation, not a claim that source relations are true object labels; SW0128's failed pseudo-instance precision remains relevant. No confidence-threshold teacher, GT tuning or RGB decoder is introduced. Exact weighting and budget would require registration after the pending outcomes, not selection now.

Any eventual candidate must improve the original native32 source and a matched control before all-seed/70k expansion. Fixed320 modal4px metrics, actual-spike masks and comparable Slot scoring remain unchanged. Conditional seed1 evidence is not all-gate necessity, three-seed superiority or positive unique-data scaling; the source already inherits70k training.
