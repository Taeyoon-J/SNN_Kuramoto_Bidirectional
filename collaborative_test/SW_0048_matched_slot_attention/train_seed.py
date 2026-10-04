"""Train official Slot Attention object-discovery model on HDF5 CLEVR IDs 0-999."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from protocol import make_epoch_balanced_batches, official_learning_rate, training_protocol


def load_model_module(model_py):
    path = Path(model_py).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("official_slot_attention_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import official model module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-py", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--decay-steps", type=int, default=2500)
    parser.add_argument("--smoke", action="store_true",
                        help="Run 2 CPU updates on 32 training images with batch size 4.")
    args = parser.parse_args()
    if args.seed < 0:
        raise ValueError("seed must be non-negative.")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(path.name != "training.log" for path in output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True)

    if args.smoke:
        steps, batch_size, num_examples = 2, 4, 32
        warmup_steps, decay_steps = 1, 2
    else:
        steps, batch_size, num_examples = args.steps, args.batch_size, 1000
        warmup_steps, decay_steps = args.warmup_steps, args.decay_steps
    if steps <= 0 or batch_size <= 0 or num_examples <= 0:
        raise ValueError("steps, batch size, and subset size must be positive.")

    protocol = training_protocol(args.seed, num_examples, batch_size, steps,
                                 warmup_steps, decay_steps)
    protocol.update({
        "dataset": str(Path(args.dataset).resolve()),
        "model_py": str(Path(args.model_py).resolve()),
        "model_sha256": hashlib.sha256(Path(args.model_py).read_bytes()).hexdigest(),
        "schedule": "lr=base*min(step/warmup,1)*decay_rate**(step/decay_steps), matching official object-discovery train.py",
        "protocol_type": "smoke_test_only" if args.smoke else "matched_data_pass_budget_candidate",
        "checkpoint_format": "TensorFlow Checkpoint(network, optimizer, global_step)",
    })
    (output_dir / "training_protocol.json").write_text(
        json.dumps(protocol, indent=2, allow_nan=False), encoding="utf-8")

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    with h5py.File(args.dataset, "r") as source:
        image_dataset = source["image"]
        if image_dataset.shape[0] < num_examples:
            raise ValueError(f"Need {num_examples} training rows; dataset has {image_dataset.shape[0]}.")
        images = image_dataset[:num_examples].astype(np.float32)
    images = images / 127.5 - 1.0
    if tuple(images.shape[1:]) != (128, 128, 3):
        raise ValueError(f"Expected HDF5 image shape [N,128,128,3], got {images.shape}.")
    batches = make_epoch_balanced_batches(num_examples, batch_size, steps, args.seed)
    model_module, model_path = load_model_module(args.model_py)
    network = model_module.build_model((128, 128), batch_size, 11, 3,
                                       model_type="object_discovery")
    optimizer = tf.keras.optimizers.Adam(4e-4, epsilon=1e-8)
    global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="global_step")
    checkpoint = tf.train.Checkpoint(network=network, optimizer=optimizer,
                                     global_step=global_step)
    manager = tf.train.CheckpointManager(checkpoint, str(checkpoint_dir), max_to_keep=3)
    log_path = output_dir / "training_loss.csv"
    started = time.time()
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "learning_rate", "reconstruction_mse", "elapsed_seconds"])
        for step in range(steps):
            lr = official_learning_rate(step, warmup_steps=warmup_steps,
                                        decay_steps=decay_steps)
            optimizer.learning_rate.assign(lr)
            batch = tf.convert_to_tensor(images[batches[step]], dtype=tf.float32)
            with tf.GradientTape() as tape:
                reconstruction, _, _, _ = network(batch, training=True)
                loss = tf.reduce_mean(tf.square(batch - reconstruction))
            gradients = tape.gradient(loss, network.trainable_weights)
            optimizer.apply_gradients(zip(gradients, network.trainable_weights))
            global_step.assign_add(1)
            writer.writerow([step + 1, lr, float(loss.numpy()), time.time() - started])
            if (step + 1) % 100 == 0 or step + 1 == steps:
                stream.flush()
                print(f"seed={args.seed} step={step + 1}/{steps} loss={float(loss.numpy()):.8f}",
                      flush=True)
        saved = manager.save(checkpoint_number=int(global_step.numpy()))
    protocol["checkpoint_prefix"] = saved
    protocol["model_py"] = str(model_path)
    (output_dir / "training_protocol.json").write_text(
        json.dumps(protocol, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "TRAINING_COMPLETED").write_text(
        f"steps={steps}\ncheckpoint={saved}\n", encoding="utf-8")
    print(f"TRAINING_COMPLETED {saved}", flush=True)


if __name__ == "__main__":
    main()
