# SW0043 - Peer-exact factorized Kuramoto dynamics

This seed-0 diagnostic ports the peer `prepare_coupling` plus factorized
matrix-multiply rollout as an opt-in backend. `pairwise` remains the default;
it preserves the existing parameters, state-dict keys, and initialization RNG
sequence. The factorized path changes the arithmetic order, so the local
tests check numerical and gradient closeness rather than bitwise output
identity. With fixed graph feedback disabled, coupling kernels are prepared
once and reused through the rollout; with feedback enabled, they are prepared
for each changed graph.

SW0043 is the peer-recipe dynamics diagnostic, distinct from SW0042's
HDF5-aligned training experiment. Training uses the peer cached gamma sequence
`/work/USERS/tkim1/gamma_sequences/wm_patch_gamma_seq_k8_grid16.pt`; evaluation
uses `/work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt` to address peer
validation IDs 6000-6999. Training uses seed 0 and the shared-projection BIM6
recipe. It does not use SW0042's HDF5-aligned gamma split. `run.sh GPU_ID 0`
trains; it refuses to overwrite an existing
checkpoint or log. `watch_and_evaluate.sh GPU_ID TRAIN_PID` waits for that
launcher, verifies the checkpoint, then runs the short and long evaluations.

Evaluate first on formal peer validation IDs 6000-6999 with peer targets:

```bash
bash collaborative_test/SW_0043_peer_exact_dynamics/evaluate.sh GPU_ID 0 short
bash collaborative_test/SW_0043_peer_exact_dynamics/evaluate.sh GPU_ID 0 long
```

Short means T256/settle64; long means T1024/settle512. Both sweep synchrony
thresholds .05, .10, .20, .35, and .50 and record peer-target metrics, with
HDF5 target scoring on the same predictions as a cross-target diagnostic.
The optional phase endpoint is an affinity diagnostic, not a classifier score.
Evaluation uses the same cached peer gamma sequence and explicitly selects
`factorized` dynamics. Target identity mapping is documented in SW0040 and
the evaluation JSON records the manifest/provenance.

`diagnose_one_batch.py` compares pairwise and factorized rollouts and a single
diagnostic optimizer update on cached gamma; it is a local backend comparison,
not a claim of a separately imported peer implementation. Run the focused
tests with `python -m pytest collaborative_test/SW_0043_peer_exact_dynamics/test_factorized.py -q`.
No training or remote evaluation is launched by this folder's tests.
The diagnostic defaults to cached training row/global image ID 0.

## Running result

Seed0's local factorized training completed all 40 epochs with logs matching
the peer exact recipe. Every learned tensor is equal to the peer original; the
only state-dict difference is `sc`. Peer stores an identity matrix (min 0,
max 1, mean 0.00390625, 256 nonzeros), while the local Pearson matrix has min
0.7572239041, max 1, mean 0.9174145460, and 65,536 nonzeros. Their maximum
absolute difference is 0.9999990463, and the current graph calculation matches
the local Pearson matrix. This recipe has `structural_weight=0`, and learned
graph forward does not use `sc`, so the buffer difference does not affect its
forward path. The serialized checkpoint files differ because `sc` differs;
the learned checkpoint state matches.

A direct same-input CPU T16 factorized comparison is bitwise equal for graph,
theta, membrane, and spikes (maximum absolute difference 0). See
`results/local_factorized_vs_peer_original.json`. This corrects the earlier
inference that pairwise/factorized arithmetic differences caused the seed0
training divergence; the exact epoch logs and same-input comparison do not
support that explanation. Formal validation metrics are now complete and
remain separate from the same-input dynamics equivalence check. On peer
validation IDs 6000-6999, the long T1024/settle512 evaluation against peer
targets scores 0.640695/0.632759/0.463078 at threshold .10 and
0.656050/0.614347/0.480345 at threshold .35 (FG-ARI / foreground IoU / matched
object IoU). These are completed factorized-checkpoint evaluations, not a
claim that all peer-recipe seeds reproduce equally.
