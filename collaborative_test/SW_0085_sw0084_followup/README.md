# SW0085: automatic SW0084 selection, activation-flow diagnosis, and promotion

SW0085 starts only after the durable SW0084 pilot queue has completed. It
selects one of the eight fixed arm/epoch checkpoints without changing a
threshold or making a per-image choice. Candidates that improve all three
pilot metrics are preferred. Within that set, the rule maximizes the worst
relative metric improvement; if none improves all three, the same maximin rule
chooses the least imbalanced candidate for diagnosis.

The selected candidate and the exact SW0072 seed1 baseline then run the SW0064
layer/time activation-flow probe on IDs1320-1351. This locates whether joint
training changes object separation first at gamma, graph, Kuramoto, gate,
dendritic, membrane, or spike. Masks are used only after forward inference for
this diagnostic and never affect predictions.

Only a checkpoint that strictly improves all three fixed pilot metrics is
automatically promoted to the full 320-image T1024/settle512 evaluation at the
already registered spike-CC threshold `.35`. Otherwise SW0085 stops after the
causal diagnostic, avoiding a costly long evaluation of a rejected direction.
The durable queue follows the same GPU policy: any idle GPU0-3, restricted to
GPU0-1 while a `tkim1` process is present.

The completed selection chose A-epoch10 for diagnosis; it scores
`.646381/.489311/.453914` against the seed1 baseline
`.708757/.468105/.509881`, so it was not promoted. Its object-separation margin
improves at gamma (`+.004869`), graph (`+.016817`), and late Kuramoto PLV
(`+.002002`), then falls at the gate (`-.057376`). The lower separation remains
at dendritic (`-.037796`), membrane (`-.034061`), and spike (`-.057376`). This
localizes the rejected joint-training direction's first material loss to the
gate transition.

