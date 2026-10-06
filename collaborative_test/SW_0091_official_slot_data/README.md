# SW0091 — official released Slot Attention data

The released Google object-discovery code loads TFDS `clevr:3.1.0`, whose
official train split contains 70,000 images. It then reserves the first 512
train rows for `train_eval`, filters training scenes to at most six objects,
applies the fixed center crop `(29:221, 64:256)`, resizes to 128x128, and trains
with seven slots.

`inspect_and_prepare.py --download` downloads and prepares the exact TFDS
release under `/Data0/kevinswk/datasets/tfds`. This is separate from our
100,000-image `clevr_10-full.hdf5` render and must remain a separate data-domain
condition.

