# SW0188: continue the actual SW0186 incumbent weights

SW0187's joint-encoder unique1024 arm did not improve FG over its contemporaneous
repeated512 control. Close that scaling recipe. This experiment tests the
previously requested alternative: retain the complete successful incumbent
instead of returning to Source097 for every new training stage.

Initialize each seed0/1/2 from its matching completed SW0186 checkpoint, including
the trained original RGB CNN, context, graph, rhythm/core and integration tensors.
Verify the entire model digest equals that incumbent's final digest on CPU and
again on GPU. Reset all five Adam groups to empty state explicitly. This is
weight continuation with a new optimizer stage, not exact Adam/RNG resumption.
Do not reset the context or any gating to its fresh Source097 values.

Keep the successful QCE objective, temperature .1, all five learning rates,
TRAIN512 twice, fixed64 proposals/B16/micro4, T1024/settle512/live-tail64,
clip1, projections and strict measured full-forward loss accept/rollback.
No additional data or gates are introduced in this test. Its purpose is whether
an additional matched adaptation budget improves the actual incumbent.
The incumbent itself already inherited full70k RGB core training; these image
counts describe the additional supervised native32 stage only.

Before deployment: compile every source, test queue/rollback/readout/scoring,
exercise the actual score producer ID at the audit entry point, and verify
independent arithmetic against actual archived score vectors. Before CUDA:
validate all three original SW0186 checkpoints/manifests, exact warm model SHA,
trained original CNN SHA, frozen preprocessing and five-family empty Adam.
First two real GPU proposals must demonstrate emitted-S and all-family credit,
full-forward loss parity and a committed actual model/CNN/Adam update.

Use the existing fixed320/native32 modal contract and unchanged spike-derived
Q/QCC/SW165 production assignment. Freeze all three seed predictions and both
conditional/standalone endpoints before reading validation GT for the new score.
Primary gate: standalone FG mean strictly above SW0186 and paired-image95% CI
lower above zero, plus all three metric means strictly above comparable Slot70k.
Report per-seed scores, both endpoints, both IoUs and FG.87 separately. An
independent audit must recompute actual masks, modal GT, seed means, intervals
and gates. Never declare unsupervised improvement or gating causality from this
supervised continuation. Native32 joint-encoder70k remains a separate goal.

One fixed experiment; no acceptance-count retry or small-hyperparameter rescue.
If the improvement gate fails, close this continuation recipe. Queue only on
exclusive idle GPUs and yield our worker when a foreign user arrives.
