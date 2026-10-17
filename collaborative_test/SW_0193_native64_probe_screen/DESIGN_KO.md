# SW0193: small native64 before/after generalization screen

The primary target remains native64 three-seed fixed320 FG-ARI>=.90, retaining
meaningful gates and reporting both IoUs. Routine development uses mini tests;
70k training requires a large replicated improvement first.

Before spending more native64 optimizer time, measure the actual transferred
incumbent and the already-trained SW0192 probe on predetermined32 development
images, canonical IDs1320..1351. This is an exploratory seed0 screen, not a
three-seed benchmark, fresh70k training, or proof of FG.90. The eight TRAIN
images and two B4 proposals are inherited from SW0192 and disclosed as such.
The larger TRAIN64/16-proposal mini stage remains a subsequent decision after
this screen and measured runtime; do not claim it already implemented.

Baseline is the exact unadapted SW0186 seed0 -> native64 mapping admitted in
SW0192. Candidate is SW0192's final, successfully admitted native64 model.
Require the SW0192 report, boundary hash, learned model hash, accepted steps,
first-two gradient credit, and current implementation fingerprint. No updates
in this experiment. Same RGB, same normalization, graph/T1024/settle512 and
production QCC plus foreground-induced merge at64x64; min_group32 preserves
the native32 min_group8 spatial area. No postprocessing or loss sweep.

Both endpoints use the SAME per-image inference seed193000+image_offset. Save
each direct native64 prediction atomically and bind model/RGB/source hashes.
Re-use only fully validated matching rows after yielding a foreign GPU owner.
Freeze both complete prediction archives and validate all three Slot64 source
archives before opening development GT. Pool canonical128 GT and Slot labels
directly by production2x2 modal. Never interpolate native32 masks or targets.

Report all three scores for baseline/candidate/Slot seed0 and Slot three-seed
mean on the same32 images. The latter is context, not a matched-seed superiority
claim. Primary screen is candidate-minus-baseline FG mean and paired-image95%CI;
all3 IoU changes are disclosed. Verify every metric independently from stored
masks using the SW0185 arithmetic, including independent modal pooling. Keep
imagewise GT-derived values private; publish aggregates and hashes only.

Positive screen: FG gain>=.01 and paired-image95%CI lower>0. This admits more
mini training or full320/three-seed confirmation; it is not final success. A
negative result closes the two-proposal adaptation as a useful performance
change; it does not prove that further native64 training cannot help. Diagnose
the initial resolution transfer and measured resource cost before more training.
