# Current native64 architecture

Run from the repository root:

```powershell
python -B collaborative_test/architecture_figures/draw_native64.py
```

The generator writes editable SVG, PDF, PNG and `provenance.json` beneath
`collaborative_testing/generated/architecture_figures/native64_current/`.
Generated files remain outside the research source tree, following the user's
artifact organization preference. Matplotlib and NumPy are required; no GPU,
checkpoint, dataset or network connection is needed.

The figure describes the SW0227 / SW0223 verified native64 reference, rather
than generic optional features of `S2NetCore`. Its execution path is the custom
SW0134 rollout. There are 4,096 patch nodes, four phase components per node,
1,024 recurrent steps, and a readout over the last 512 steps. The graph is
input-conditioned and constant within one rollout. Frozen graph parameters
do not imply that different images share an identical graph.

Blue blocks indicate the encoder and actual 8-to-4 gamma projection trained
in the latest teacher adaptation. Graph and remaining core parameters are
frozen in that adaptation, not necessarily throughout their checkpoint
lineage. Local previous-spike input and membrane reset are present. Spike
feedback to Kuramoto and synchrony feedback to the graph are disabled in this
reference; the figure does not establish the causal benefit of any gate.

The illustration is a schematic, including the input scene. It contains no
measured activation panels or sample predictions. Scores are preserved at
full precision in the generated provenance, separate from the architecture.
They concern fixed320 reused validation, three-seed means, and do not establish
an untouched-test result or matched training supervision with Slot Attention.

The source paths underlying the diagram are listed in `provenance.json`.
Visual inspection of the rendered PNG checked text fit, arrows, feedback
labels and the separation of inference from training and evaluation.
