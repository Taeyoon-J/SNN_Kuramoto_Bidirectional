# Slot Attention data provenance

The paper experiment, the released Google checkpoint, and SW0056 are three
different training-data conditions and must not be described as one baseline.

## NeurIPS 2020 paper

For unsupervised object discovery, Locatello et al. used `CLEVR (with masks)`,
Multi-dSprites, and Tetrominoes. The CLEVR experiment used the first 70,000
samples of the DeepMind multi-object `CLEVR (with masks)` dataset and cropped
images to emphasize the center. CLEVR6 restricted scenes to at most six objects,
used seven slots and three Slot Attention iterations, and reported foreground
ARI over five seeds. Object-discovery evaluation used 320 examples.

The paper's separate supervised set-prediction experiment used original CLEVR,
with 70,000 training and 15,000 validation images containing three to ten
objects. These set-prediction data and metrics are not the object-discovery
baseline used here.

Primary source: Locatello et al., *Object-Centric Learning with Slot Attention*,
NeurIPS 2020, Section 4 and Table 1:
<https://proceedings.neurips.cc/paper_files/paper/2020/file/8511df98c02ab60aea1b2356c013bc0f-Paper.pdf>.

## Released Google code and checkpoint

The subsequently released object-discovery code uses the original TFDS CLEVR
dataset with `max_n_objects=6`, a center crop, seven slots by default, three
iterations, batch64, and 500,000 training steps. The accompanying repository
explicitly warns that this differs from the paper's DeepMind
`multi_object_datasets` render and that their statistics are not directly
comparable. The public `gs://gresearch/slot-attention/object-discovery/ckpt-500`
used in SW0046 belongs to this released-code condition, not to our HDF5 IDs.

Primary code:
<https://github.com/google-research/google-research/blob/master/slot_attention/object_discovery/train.py>.

## SW0056 matched-data comparator

SW0056 trains the released Slot Attention architecture from scratch on exactly
the same 2,500 HDF5 training scenes used by the current SNN candidate: IDs0-999
and 1640-3139. Each scene appears ten times, giving 25,000 exposures and 1,563
updates per seed. Seeds0/1/2 use independent initialization and data order, then
score on aligned HDF5 validation IDs1320-1639 through the fixed patch contract.

This removes the data-domain mismatch and supplies the requested matched-budget
comparison. It is not a 500,000-step reproduction of the published training
budget. Convergence and the shorter budget must remain explicit when interpreting
its result.

