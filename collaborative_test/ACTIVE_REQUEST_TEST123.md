# Current authorized goal scope

User instruction on 2026-10-06: run only experiments 1, 2 and 3 to completion,
then pause goal mode. Suspend architecture/loss/model modification experiments.

- Experiment 1 = SW0090: our model on our 70,000 unique HDF5 training images.
- Experiment 2 = SW0092 our_on_official: our fixed-encoder/frozen-graph recipe
  trained on official released Slot CLEVR6 after official crop/filter (34,766
  images). Existing encoder/graph pretraining provenance must be disclosed.
- Experiment 3 = SW0092 slot_our70000: official Slot architecture from scratch
  on the exact same 70,000 HDF5 training IDs as experiment 1.

Complete all three seeds and registered epoch1/3/10 evaluations on held-out
HDF5 IDs1320-1639. Finish experiment 1 three-image mask visualizations. Retrieve
results locally, record comparison and push the collaboration branch. Only
after those deliverables are complete, call update_goal(status="paused") as
explicitly requested by user. Do not initiate further model improvements.

Use GPUs0-3 while available. Respect existing tkim1 ownership and earlier
instruction to return to GPU0-1 if that user occupies GPUs. Avoid stopping any
unrelated process. Do not pause just while waiting for authorized jobs.

Latest verified state: all experiments1/2/3 are complete and archived locally.
All 27 full-320 evaluations and experiment1's three-image mask visualizations
were checked. Final comparison is in `TEST123_FINAL_RESULTS.md` and
`SW_0092_cross_dataset_training/results/test123_final_comparison.json`.
Experiment3 final epoch10 mean is .774933/.203589/.206937; the earlier .853923
FG-ARI mean used only seeds1/2 and is superseded by the three-seed result.
Next action after sharing final artifacts: pause goal mode as explicitly
requested. Do not start architecture/loss experiments or further training.
