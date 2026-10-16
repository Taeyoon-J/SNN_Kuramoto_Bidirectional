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

Training completion, prediction, and fixed320 evaluation results are not yet available.
This is a mask-supervised experiment, not an unsupervised leaderboard update,
fresh native32/70k training, or evidence that individual gates are necessary.
SW0165 remains the unsupervised incumbent.
