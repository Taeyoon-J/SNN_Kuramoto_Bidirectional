# SW0053 - low-learning-rate early-stop matrix

SW0052 selected the no-diversity, LR `3e-4`, epoch-25 seed2 checkpoint. Its
long `.35` result `.395707/.189112/.159085` strictly dominates the same run at
epoch 40. SW0053 separates early stopping from learning rate and tests whether
the selected low-LR recipe harms seeds 0/1.

Three predeclared 25-epoch runs use the same aligned training gamma and BIM6
architecture/loss:

| GPU arm | Seed | LR | Purpose |
|---|---:|---:|---|
| baseline early stop | 2 | 1e-3 | isolate epoch 25 without low LR |
| low-LR expansion | 0 | 3e-4 | seed0 harm/generalization check |
| low-LR expansion | 1 | 3e-4 | seed1 harm/generalization check |

Each run starts only after the SW0050 real-gamma one-update preflight is
current for the exact trainer/run code, refuses existing output, trains for 25
epochs, verifies the checkpoint, and evaluates both fixed short and long HDF5
windows. Ground truth is used only for scoring. No test split is used.

```bash
bash collaborative_test/SW_0053_lr_early_stop_matrix/launch_parallel.sh 1 2 3
```

## Results

All three arms completed training plus short/long evaluation. At the common
long threshold `.35`:

| Run | FG-ARI | Foreground IoU | Matched-object IoU |
|---|---:|---:|---:|
| seed0, low LR, epoch25 | .700964 | .613168 | .484189 |
| seed1, low LR, epoch25 | .583643 | .343275 | .404509 |
| seed2, low LR, epoch25 (SW0052) | .395707 | .189112 | .159085 |
| **low-LR three-seed mean** | **.560105** | **.381852** | **.349261** |
| seed2, LR1e-3, epoch25 control | .345872 | .172243 | .126184 |

At threshold `.50`, the low-LR three-seed mean is
`.562902/.381328/.356857`. Compared with the prior SW0042 `.35` mean
`.415879/.418226/.296118`, low LR plus epoch25 greatly improves FG-ARI and
object IoU, while foreground IoU trades down but remains above the audited
single-checkpoint Slot reference. Seed2 low LR strictly exceeds the matched
LR1e-3 epoch25 control on all three metrics, so the rescue is not explained by
early stopping alone. Seed0 improves all three metrics; seed1 gains ARI and
object IoU but loses foreground IoU. Full outputs are under `results/`.
