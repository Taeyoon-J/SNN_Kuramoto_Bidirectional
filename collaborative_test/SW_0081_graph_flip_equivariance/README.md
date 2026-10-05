# SW0081: graph-only horizontal-flip equivariance

This experiment asks whether a frozen image encoder can provide a useful
geometric consistency target for the SW0072 seed1 learned graph. It starts from
the full SW0072 seed1 core, keeps the registered RGB encoder and preprocessing
statistics frozen, and updates only `core.graph_generator`.

For each training image, the graph generator produces a detached teacher from
the registered cached gamma. The same frozen encoder processes a horizontal
reflection of the RGB image; its standardized, clipped features are pooled to
the 16x16 patch grid and converted to gamma. The graph on that gamma is mapped
back to source coordinates using the exact column-reversal permutation. The
equivariance term is MSE between aligned flipped adjacency and detached teacher
adjacency. Existing phase binding loss plus 5x component-spike synchrony loss
remain active. No masks or object counts are loaded.

Training examples are the registered 2500 rows: HDF5 IDs 0-999 followed by
1640-3139, paired in order with the SW0055 cached gamma. Encoder weights and
stats come from the registered 2026-10-02 source run. The 5-epoch seed1 arms
use graph LR `3e-5` and differ only in equivariance weight.

The real one-update preflight measures raw equivariance MSE and separate graph
gradient norms for binding-plus-spike and equivariance on the same batch. It
derives weights that target equivariance gradient contributions of 0.1x and
1x the binding-plus-spike gradient; the arm weights are read from that immutable
preflight artifact rather than guessed. The preflight also verifies finite
losses, nonzero graph update, bitwise unchanged non-graph core and encoder, and
records all source/code hashes. The measured MSE and gradient scale are saved
in its manifest. No real-asset training or evaluation has been launched here.

Run preflight on an idle GPU0 or GPU1:

```bash
bash collaborative_test/SW_0081_graph_flip_equivariance/preflight.sh 0
```

After it succeeds, launch both arms on GPUs0/1:

```bash
bash collaborative_test/SW_0081_graph_flip_equivariance/launch_pair.sh
```

Evaluate each checkpoint and the fixed SW0072 seed1 baseline on aligned IDs
1320-1351 with T256/settle64, vth `.06`, and spike connected-components at
threshold `.35`. Scoring labels are opened only by the evaluator after
predictions are formed:

```bash
bash collaborative_test/SW_0081_graph_flip_equivariance/evaluate.sh 0 baseline
bash collaborative_test/SW_0081_graph_flip_equivariance/evaluate.sh 0 w0p1
bash collaborative_test/SW_0081_graph_flip_equivariance/evaluate.sh 1 w1x
/Data0/kevinswk/envs/snn/bin/python collaborative_test/SW_0081_graph_flip_equivariance/compare_pilot.py \
  --baseline trained_models/SW0081_fixed_pilot/baseline_seed1_n32.json \
  --w0p1 trained_models/SW0081_fixed_pilot/w0p1_seed1_n32.json \
  --w1x trained_models/SW0081_fixed_pilot/w1x_seed1_n32.json \
  --output trained_models/SW0081_fixed_pilot/pilot_gate.json
```

The pilot gate advances an arm only if FG-ARI, foreground IoU, and matched
object IoU all strictly improve over the fixed SW0072 seed1 result. It does not
select thresholds or use validation labels to set equivariance weights. These
are seed1 pilot comparisons, not multi-seed claims.

Server syntax checks and four focused unit tests passed. The real one-update
preflight measured raw flip-equivariance MSE `1.10666e-4`, binding/spike graph
gradient norm `30.2575`, and raw equivariance gradient norm `9.22606e-4`.
This fixes the 0.1x/1x arm weights at `3279.57` and `32795.68`. The update kept
the encoder and every non-graph core tensor bitwise unchanged and changed the
graph by `3.0041e-5`. Both five-epoch arms were launched on GPUs0/1.
