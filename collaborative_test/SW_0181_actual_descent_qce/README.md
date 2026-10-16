# SW0181 — actual-loss acceptance for QCE training

SW0180 found that seed 2 deteriorated on the same eight TRAIN images after
unconditional QCE updates: CE 0.526248 → 0.936206 and standalone FG-ARI
0.784225 → 0.636125. Seeds 0 and 1 improved on that replay. This small diagnostic
supports testing optimization stability; it does not establish a population-wide
cause or prove that minibatch descent will improve segmentation.

SW0181 retains the source, full forward, QCE objective, four optimizer families,
learning rates, shared TRAIN512 order, and 64-proposal budget. It commits an Adam
proposal only when the same batch and RNG produce a strictly lower actual loss.
Otherwise it restores model and Adam state and consumes the batch without a retry.
All three seeds receive the same policy. See [the preregistered design](DESIGN_KO.md).

## Verified evidence

- The four transaction unit tests passed on Frontier.
- On real seed 0/1/2 model copies, both empty and populated Adam states and model
  parameters/buffers restored exactly after synthetic perturbations. Original
  checkpoint files remained unchanged.
- This CPU proof used synthetic gradients only. It performed no spike forward,
  committed no training updates, initialized no CUDA, and read no validation data.
- Receipt: [CPU transaction proof](results_archive/cpu_v1/transaction_cpu_receipt.json).

The server's full 16-test suite and an actual subprocess CLI check passed.
An earlier undefined seed-choice constant stopped the first supervisor before
queue creation or training; the failure log and corrected-file hashes are
preserved in [the deployment receipt](results_archive/cpu_v2/training_cpu_receipt.json).

All three actual GPU workers started on GPUs 1/2/3. The first two proposals per
seed passed emitted-spike and four-family gradient checks. Same-RNG full no-grad
before loss differed from the gradient-forward value by at most 4.66e-9, below
the fixed acceptance guard. Both acceptance and rejection occurred, and actual
serialized Adam steps matched the accepted count independently of the proposal
cursor. This proves that the policy is executing, not that segmentation improved.

All three training runs completed and their model/optimizer/history artifacts
passed validation. Each seed consumed the registered 64 proposals, with 32/38/34
accepted updates for seeds 0/1/2 respectively. The remaining proposals were
rejected without replacement. This is fewer committed Adam steps than the
unconditional control, with the same proposal budget and unique TRAIN512 images.
Each run additionally performed 64 post checks and two pre checks, for 130
logical B16 full forwards (2,080 image forwards) before any prediction.
See [training records](results_archive/train_v1/seed0_training_record.json).

Fixed320 prediction and evaluation results are not yet available.
Three prediction supervisors and the GT-last score coordinator are now queued;
all four processes were observed live. Each prediction waits for its own completed
training artifact, and scoring waits for all three immutable predictions.
The deployed evaluation suite passed 32 tests and four actual subprocess CLI checks.
The supplemental unconditional-QCE comparator passed two additional tests and
validated the actual pinned SW0179 score archive without reading validation masks.
Its matched-seed comparison will be run after the SW0181 score is frozen.

See [evaluation deployment verification](results_archive/eval_v1/cpu_receipt.json)
and [queue launch record](results_archive/eval_v1/launch.json).

This is a mask-supervised experiment, not an unsupervised leaderboard update,
fresh native32/70k training, or evidence that individual gates are necessary.
SW0165 remains the unsupervised incumbent.
