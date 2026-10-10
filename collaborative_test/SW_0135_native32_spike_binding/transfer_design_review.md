# Separate native32 transfer screen from completed SW0134

Prospective Sol6.1 review, 2026-10-10. **Recommend this evaluation-only screen before committing to two new4096-update native32 arms.** Keep SW0135's source97 training protocol and running parity queue unchanged. A separate registered transfer task must record its own initialization, implementation and prediction hashes. No new16 test or training is needed.

## What can transfer exactly

SW0134 `train.py` saves `checkpoint.pt` containing `wrapped_state_dict`, `encoder_state_dict`, `binder_state_dict`, `decoder_state_dict`, source SHA, seed, arm and lambda. The two completed seed1 arms are available:

- actual_joint checkpoint SHA `7bb7efcaf3640c6de3ca4ac5b8f8f265f16f6e75a2ffb24b0515862930e49beb`.
- actual_frozen checkpoint SHA `361b426179a4eb3f2d39169398557e0b898c995afab62c68cdcb025677b783ed`.

These are chosen because both registered actual-spike arms completed, not because of a new validation ranking. The interrupted gate_joint has no completed checkpoint; no three-arm/gate-usefulness conclusion is available.

Validate each immutable completion marker, manifest/checkpoint/history hashes,4096 update sequence,16 passes and original source/preflight/optimizer provenance using the original frozen134 validator. Preserve all historical artifacts. If those bindings do not validate, do not infer completion merely from a checkpoint file.

Split the wrapped state **explicitly** into `core.*` native tensors and exactly the three learned integration tensors `a_d`, `a_m`, `b`. Strip only the registered `core.` prefix, map the native tensors with the declared135 spatial2x2 conversion, strict-load the1024 core, and strict-load the learned12 integration scalars without resetting them to zero. Carry every learned graph/native/shared parameter; no partial-load fallback or source97 replacement of inconvenient tensors.

Strict-load each arm's trained encoder and the original registered feature statistics; recompute genuine gamma `[B,8,1024]` from native128 RGB with adaptive pooling32. Strict-load the entire binder state, including **the persistent134 initial_noise buffer**, learned shared slot mu/log-scale, projection2048->64, attention/GRU/MLP and normalization weights. The patch count is not a learned dimension, so these weight shapes are compatible with the135 binder. Initializing a135 head without replacing its noise buffer would change the trained classifier. Strict-load decoder weights too; they are shape-compatible, although RGB decoding is unnecessary to generate masks.

## Genuine32 execution

Use the independently implemented32 binder,1024 patch normalization, original11 slots/three refinements and full actual `[B,4,1024,512]` settled spike input. No16 feature/prediction expansion. Use the same prospectively registered32 physical foundation: K1024, top_k128, grid spacing.5, corrected log4 geodesic/full-node reduction and1024 SC. These adaptations change input dynamics; strict weight transfer does not promise exact equivalence to the16 function.

If rendering RGB diagnostics, use PATCH4 relative geometry and intra-cell variance `.00030517578125`; carry the original trained decoder rather than warm/retrain it. At evaluation put all modules in eval/no-grad mode, explicitly as inference, with full T1024/settle512. Do not treat inference freezing as evidence that a later native32 training path has correct gradients. Start batch1 to minimize both memory and batch-induced numerical changes. Hash the complete converted state and source/mapping/configuration before prediction.

One no-GT real32 batch establishes strict load, finite actual traces, genuine1024 P and output32 labels; then evaluate **both** transferred arms on all320 IDs1320..1639, freezing predictions before GT. Primary is the learned32 P readout with its frozen background/tie rule; report native32 QCC separately using.50/min8. Include the same-seed mapped-source97 QCC reference if not already available from a correctly bound native32 evaluation. Score all three metrics with the registered4px modal GT and report the fixed comparable Slot32 baseline. No held-out reserve access, checkpoint selection, slot/temperature/geometry tuning or automatic training extension.

## Interpretation

This screen can cheaply identify whether the already learned classifier transfers useful instance information at32, and whether joint versus frozen **16-training history** transfers differently. It cannot establish a native32 joint-training causal effect, native32 data scaling, all-gate necessity, a fresh70k training comparison or three-seed success. Resolution, graph adaptation and larger patch competition can degrade a useful16 head; a transfer failure does not itself reject the fresh135 recipe. Conversely, a promising single seed does not amend135 or satisfy the final objective: any native32 fine-tuning/three-seed extension needs its own prospective protocol and preserved pretrained exposure accounting.
