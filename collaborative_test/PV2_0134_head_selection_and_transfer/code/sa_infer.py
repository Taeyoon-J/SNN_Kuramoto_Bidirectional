#!/usr/bin/env python3
"""Run the published Slot Attention checkpoint on dumped frames and save the RAW slot map.

Slot Attention at our grid scores Foreground IoU 0.2022 against a truth foreground of
0.2115, calling 85.1% of the image foreground. A model that calls four times too much of the
image an object is not being read correctly, and the one free choice in reading it is which
slot counts as background: slot ids are arbitrary, so the contract's background id 0 has to
be assigned by a rule.

So the slot map is saved before any remapping, and `sa_bg.py` chooses the rule on the
validation split -- never on fixed320 -- exactly as the contract requires of a threshold.

The frames come from `dataset.rgb`, already 128x128, the resolution Slot Attention was
published at and the same frames our core is given. There is no crop and no resize, so grid
cell (r, c) means the identical pixels in the prediction and in the target.
"""
import argparse, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--frames', required=True); ap.add_argument('--checkpoint-dir', required=True)
ap.add_argument('--model-dir', required=True); ap.add_argument('--grid', type=int, default=64)
ap.add_argument('--num-slots', type=int, default=7); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
a = ap.parse_args()

blob = np.load(a.frames); images, ids = blob['rgb'], blob['ids']
print(f"{blob['split']}: {len(ids)} frames", flush=True)
import tensorflow as tf
for dev in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(dev, True)
# Slot Attention resamples its slots from slots_mu + exp(slots_log_sigma)*normal on every
# forward pass, so inference is stochastic; seeding makes the run reproducible.
tf.random.set_seed(a.seed); np.random.seed(a.seed)
sys.path.insert(0, a.model_dir)
import model as sa_model
net = sa_model.build_model(resolution=(128,128), batch_size=1, num_slots=a.num_slots,
                           num_iterations=3, model_type='object_discovery')
ckpt = tf.train.Checkpoint(network=net)
manager = tf.train.CheckpointManager(ckpt, a.checkpoint_dir, max_to_keep=1)
ckpt.restore(manager.latest_checkpoint)
print('restored from', manager.latest_checkpoint, flush=True)

GRID, ph = a.grid, 128 // a.grid
slots = np.zeros((len(ids), GRID, GRID), dtype=np.int64)
for i, img in enumerate(images):
    x = img.astype(np.float32)/255.*2. - 1.            # the published scaling, [-1, 1]
    _, _, masks, _ = net(x[None])
    assign = np.argmax(np.asarray(masks)[0, ..., 0], axis=0)
    tiles = assign.reshape(GRID, ph, GRID, ph).transpose(0,2,1,3).reshape(GRID, GRID, -1)
    slots[i] = np.apply_along_axis(
        lambda t: np.bincount(t, minlength=a.num_slots).argmax(), 2, tiles)
    if (i+1) % 80 == 0: print(f'  {i+1} frames', flush=True)

sizes = np.stack([np.bincount(s.reshape(-1), minlength=a.num_slots) for s in slots])
np.savez(a.out, slots=slots, ids=ids, num_slots=a.num_slots, split=str(blob['split']))
share = sizes.sum(0)/sizes.sum()
print('\nshare of all patches by slot: ' + '  '.join(f'{k}:{v:.3f}' for k, v in enumerate(share)))
print('slots used per image, mean:', float((sizes > 0).sum(1).mean()))
print('wrote', a.out, flush=True)
