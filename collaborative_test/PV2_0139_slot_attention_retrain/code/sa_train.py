#!/usr/bin/env python3
"""Retrain Slot Attention on our render, so the reference column is a matched comparison.

No training loop ships with what we have -- only `model.py` and `utils.py` -- so this is the
published object-discovery recipe written out: L2 reconstruction, Adam, linear warmup then
exponential decay, slots resampled every forward pass.

  base lr 4e-4, batch 64, 7 slots, 3 iterations, warmup 10,000 steps,
  decay rate 0.5 every 100,000 steps

`--steps` is deliberately a flag rather than a constant. The published budget is 500,000 steps,
which is days on a TITAN RTX; a shorter run is a legitimate reference as long as the row says
how long it trained. What is NOT legitimate is a shorter run reported as "Slot Attention",
so the step count, the exposure count and the wall time all go into the sidecar json and must
appear in the reported row.

Checkpoints are written every `--save-every` steps so the run can be scored at several budgets
without retraining, and so an interrupted run resumes instead of restarting.
"""
import argparse, json, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--frames', required=True, help='npy written by sa_dump_train.py')
ap.add_argument('--model-dir', required=True, help='directory holding model.py and utils.py')
ap.add_argument('--out', required=True, help='checkpoint directory for this run')
ap.add_argument('--steps', type=int, required=True)
ap.add_argument('--batch', type=int, default=64)
ap.add_argument('--num-slots', type=int, default=7)
ap.add_argument('--num-iterations', type=int, default=3)
ap.add_argument('--lr', type=float, default=4e-4)
ap.add_argument('--warmup', type=int, default=10000)
ap.add_argument('--decay-rate', type=float, default=0.5)
ap.add_argument('--decay-steps', type=int, default=100000)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--save-every', type=int, default=10000)
a = ap.parse_args()

import sys
sys.path.insert(0, a.model_dir)
import tensorflow as tf
for dev in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(dev, True)
import model as sa_model
import utils as sa_utils

frames = np.load(a.frames, mmap_mode='r')
n = len(frames)
print(f'{n} training frames, {a.steps} steps at batch {a.batch} '
      f'= {a.steps*a.batch} exposures ({a.steps*a.batch/n:.1f} epochs)', flush=True)

tf.random.set_seed(a.seed); np.random.seed(a.seed)
net = sa_model.build_model(resolution=(128, 128), batch_size=a.batch,
                           num_slots=a.num_slots, num_iterations=a.num_iterations,
                           model_type='object_discovery')
optimizer = tf.keras.optimizers.Adam(a.lr, epsilon=1e-8)
step = tf.Variable(0, trainable=False, dtype=tf.int64)
ckpt = tf.train.Checkpoint(network=net, optimizer=optimizer, step=step)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
manager = tf.train.CheckpointManager(ckpt, str(out), max_to_keep=50)
if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    print(f'resumed from {manager.latest_checkpoint} at step {int(step)}', flush=True)


def learning_rate(s):
    """Linear warmup then exponential decay, as published.

    Written with plain tf ops so it is safe both inside and outside the traced step. A zero
    warmup is allowed and means "no warmup", rather than dividing by zero.
    """
    f = tf.cast(s, tf.float32)
    base = tf.constant(a.lr, tf.float32)
    if a.warmup > 0:
        base = base * tf.minimum(f / float(a.warmup), 1.0)
    return base * a.decay_rate ** (f / float(a.decay_steps))


@tf.function
def train_step(images, lr):
    # .assign, not rebinding: in TF 2.15 the optimizer's learning_rate is a tf.Variable, and
    # assigning a tensor to the attribute inside a traced function does not take effect.
    optimizer.learning_rate.assign(lr)
    with tf.GradientTape() as tape:
        recon_combined, recons, masks, slots = net(images, training=True)
        loss = sa_utils.l2_loss(images, recon_combined)
    grads = tape.gradient(loss, net.trainable_weights)
    optimizer.apply_gradients(zip(grads, net.trainable_weights))
    return loss


# Reseeded per session, so a resumed run repeats an earlier data order. Harmless for a
# randomly sampled batch, and recorded here rather than hidden.
rng = np.random.RandomState(a.seed)
started = time.time(); losses = []
start_step = int(step)
log = out/'training.jsonl'
while int(step) < a.steps:
    idx = rng.randint(0, n, size=a.batch)
    batch = frames[np.sort(idx)].astype(np.float32)/255.*2. - 1.   # the published scaling
    loss = float(train_step(tf.convert_to_tensor(batch), learning_rate(step)))
    step.assign_add(1); s = int(step); losses.append(loss)
    if s % 100 == 0:
        el = time.time()-started
        with open(log, 'a') as f:
            f.write(json.dumps(dict(step=s, loss=float(np.mean(losses[-100:])),
                                    lr=float(learning_rate(step)), seconds=el))+'\n')
        if s % 1000 == 0:
            rate = el/max(s - start_step, 1)      # this session's steps, not all sessions'

            print(f'  step {s}/{a.steps}  loss {np.mean(losses[-100:]):.5f}  '
                  f'{rate:.3f} s/step  {(a.steps-s)*rate/3600:.1f} h left', flush=True)
    if s % a.save_every == 0 or s == a.steps:
        manager.save(checkpoint_number=s)
        print(f'  saved checkpoint at step {s}', flush=True)

manager.save(checkpoint_number=int(step))
(out/'provenance.json').write_text(json.dumps(dict(
    model='slot attention, object_discovery, retrained on clevr_with_masks',
    frames=a.frames, train_frames=n, steps=int(step), batch=a.batch,
    exposures=int(step)*a.batch, epochs_equivalent=int(step)*a.batch/n,
    num_slots=a.num_slots, num_iterations=a.num_iterations, base_lr=a.lr,
    warmup=a.warmup, decay_rate=a.decay_rate, decay_steps=a.decay_steps, seed=a.seed,
    seconds=time.time()-started,
    published_budget_steps=500000,
    note='a budget below 500,000 steps is a legitimate reference only if the reported row '
         'states the step count; never report a short run simply as "Slot Attention"'),
    indent=1)+'\n')
print(f'SA_TRAIN DONE seed {a.seed}, {int(step)} steps, '
      f'{(time.time()-started)/3600:.1f} h -> {out}', flush=True)
