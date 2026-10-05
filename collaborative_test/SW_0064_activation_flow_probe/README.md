# SW0064 - layer and time-resolved activation-flow probe

SW0064 measures the paper's causal path directly on fixed validation images:
`gamma -> learned graph -> Kuramoto/PLV -> delayed gate -> dendritic state ->
membrane -> spike`. It compares the best and worst SW0055 seeds under identical
data and hyperparameters to locate where seed instability first appears.

For each stage, ground-truth masks are read only after the forward pass. The
probe reports mean affinity for pairs of patches from the same foreground
object, different foreground objects, and foreground versus background. The
primary diagnostic is the same-minus-different margin. Dynamic stages are
measured in early, middle, and late rollout windows, showing whether object
structure strengthens or decays as activation moves through the core.

The probe changes neither prediction nor model state. It is diagnostic evidence,
never a scored model result. The initial contract is seed0 versus seed2,
IDs1320-1351, T256, three 64-step windows, native validation gamma, and the
fixed SW0055 checkpoints.

## Result

The probe completed on 32 fixed images. The same-object minus different-object
margin starts from the identical gamma value (`.0360`) but diverges most at the
graph-to-Kuramoto transition: late PLV is `.8167` for seed0 and `.5167` for
seed2. Dendritic-to-membrane propagation is nearly lossless (`.5947 -> .5917`
and `.4676 -> .4576`), while membrane-to-spike is a smaller secondary loss.

This localizes the primary seed instability to the learned graph's interaction
with Kuramoto dynamics. The next experiment therefore isolates the graph
generator instead of changing downstream gates or readouts.
