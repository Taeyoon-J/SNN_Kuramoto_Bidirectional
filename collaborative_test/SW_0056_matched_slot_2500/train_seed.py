"""Train the SW0056 2500-scene Slot Attention candidate."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

EXECUTION_BACKEND = "gpu_opt_in" if os.environ.get("SW0056_ALLOW_GPU") == "1" else "cpu_forced"
if EXECUTION_BACKEND == "cpu_forced":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import h5py
import numpy as np

from protocol import (exact_exposure_batches, official_learning_rate,
                      stable_weight_signature, training_ids, training_protocol)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_model(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("official_slot_attention_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official model at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-py", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true", help="two updates over 24 real training images, including an 8-row final batch")
    args = parser.parse_args()
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True)

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    all_ids = training_ids()
    repeats = 1 if args.smoke else 10
    if args.smoke:
        all_ids = all_ids[:24]
    batches = exact_exposure_batches(ids=all_ids, repeats=repeats, batch_size=16, seed=args.seed)
    expected_batches = 2 if args.smoke else 1563
    expected_tail = 8
    if len(batches) != expected_batches or len(batches[-1]) != expected_tail:
        raise AssertionError(f"unexpected batch plan: {len(batches)} updates, final {len(batches[-1])}")
    if not args.smoke:
        observed, counts = np.unique(np.concatenate(batches), return_counts=True)
        if not np.array_equal(observed, training_ids()) or not np.all(counts == 10):
            raise AssertionError("full run must cover every selected training image ten times")

    with h5py.File(args.dataset, "r") as h5:
        ds = h5["image"]
        if ds.shape[0] < 3140 or tuple(ds.shape[1:]) != (128, 128, 3):
            raise ValueError(f"unexpected HDF5 image array shape: {ds.shape}")
        images = np.empty((len(all_ids), 128, 128, 3), dtype=np.float32)
        image_digest = hashlib.sha256()
        for start in range(0, len(all_ids), 64):
            end = min(start + 64, len(all_ids))
            raw = ds[all_ids[start:end]]
            image_digest.update(np.ascontiguousarray(raw).tobytes())
            images[start:end] = raw.astype(np.float32)
    images = images / 127.5 - 1.0
    row_for_id = {int(image_id): row for row, image_id in enumerate(all_ids)}

    model, model_path = load_model(args.model_py)
    # The official model's unstack_and_split uses image.shape[0] as a static
    # reshape dimension. Build fixed 16- and 8-example signatures instead of
    # padding or dropping the true final batch. Both models share identical
    # weight shapes; the auxiliary model supplies gradients for the same
    # primary variables and optimizer state.
    network = model.build_model((128, 128), 16, 11, 3, model_type="object_discovery")
    auxiliary_network = None
    optimizer = tf.keras.optimizers.Adam(4e-4, epsilon=1e-8)
    global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="global_step")
    ckpt = tf.train.Checkpoint(network=network, optimizer=optimizer, global_step=global_step)
    manager = tf.train.CheckpointManager(ckpt, str(checkpoint_dir), max_to_keep=2)
    local_code = Path(__file__).resolve()
    protocol_file = Path(__file__).with_name("protocol.py")
    run_script = Path(os.environ.get("SW0056_RUN_SCRIPT", Path(__file__).with_name("run_seed.sh"))).resolve()
    dataset_stat = Path(args.dataset).stat()
    if args.smoke:
        base_protocol = {
            "seed": args.seed,
            "training_id_segments_inclusive": [[0, 23]],
            "validation_ids_inclusive": [1320, 1639],
            "unique_training_images": 24,
            "exposures_per_image": 1,
            "effective_image_exposures": 24,
            "optimizer_updates": 2,
            "full_batch_updates": 1,
            "final_batch_size": 8,
            "protocol_type": "smoke_test_only",
            "batch_policy": "16 real examples then 8 real examples using auxiliary static-batch model; no padding; smoke only",
        }
    else:
        base_protocol = training_protocol(args.seed)
    base_protocol.update({
        "dataset": str(Path(args.dataset).resolve()),
        "dataset_size_bytes": dataset_stat.st_size,
        "dataset_mtime_ns": dataset_stat.st_mtime_ns,
        "selected_training_images_sha256": image_digest.hexdigest(),
        "selected_training_ids_sha256": hashlib.sha256(all_ids.astype("<i8").tobytes()).hexdigest(),
        "model_py": str(model_path),
        "model_sha256": sha256(model_path),
        "trainer_sha256": sha256(local_code),
        "execution_backend": EXECUTION_BACKEND,
        "batch_policy_code_sha256": sha256(protocol_file),
        "run_script_sha256": sha256(run_script),
        "checkpoint_source": "scratch initialization; no pretrained weights restored",
        "checkpoint_format": "TensorFlow Checkpoint(network, optimizer, global_step)",
    })
    protocol_path = out / "training_protocol.json"
    protocol_path.write_text(json.dumps(base_protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    loss_path = out / "training_loss.csv"
    started = time.time()
    starting_weights = [np.asarray(w).copy() for w in network.get_weights()] if args.smoke else None
    finite_losses = True
    partial_batch_used = False
    partial_primary_weights_changed = False
    with loss_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "batch_size", "learning_rate", "reconstruction_mse", "elapsed_seconds"])
        for step, global_batch_ids in enumerate(batches):
            rows = np.asarray([row_for_id[int(image_id)] for image_id in global_batch_ids], dtype=np.int64)
            batch = tf.convert_to_tensor(images[rows], dtype=tf.float32)
            if int(tf.shape(batch)[0]) != len(global_batch_ids):
                raise AssertionError("input batch unexpectedly padded or truncated")
            active_network = network
            if len(global_batch_ids) != 16:
                if auxiliary_network is None:
                    auxiliary_network = model.build_model(
                        (128, 128), len(global_batch_ids), 11, 3,
                        model_type="object_discovery")
                    if len(network.trainable_weights) != len(auxiliary_network.trainable_weights):
                        raise RuntimeError("static batch signatures have different variable counts")
                    if stable_weight_signature(network.trainable_weights) != stable_weight_signature(auxiliary_network.trainable_weights):
                        raise RuntimeError("static batch signatures differ in ordered names/shapes")
                auxiliary_network.set_weights(network.get_weights())
                active_network = auxiliary_network
                partial_batch_used = True
            lr = official_learning_rate(step)
            optimizer.learning_rate.assign(lr)
            with tf.GradientTape() as tape:
                reconstruction, _, _, _ = active_network(batch, training=True)
                loss = tf.reduce_mean(tf.square(batch - reconstruction))
            grads = tape.gradient(loss, active_network.trainable_weights)
            if any(grad is None for grad in grads):
                raise RuntimeError("missing gradient in official Slot Attention model")
            if not math.isfinite(float(loss.numpy())):
                finite_losses = False
                raise FloatingPointError("non-finite reconstruction loss")
            if any(not bool(tf.reduce_all(tf.math.is_finite(
                    grad.values if isinstance(grad, tf.IndexedSlices) else grad))) for grad in grads):
                raise FloatingPointError("non-finite gradient")
            before_partial_update = ([np.asarray(w).copy() for w in network.get_weights()]
                                     if len(global_batch_ids) != 16 and args.smoke else None)
            optimizer.apply_gradients(zip(grads, network.trainable_weights))
            if before_partial_update is not None:
                partial_primary_weights_changed = any(
                    not np.array_equal(before, after)
                    for before, after in zip(before_partial_update, network.get_weights()))
            global_step.assign_add(1)
            writer.writerow([step + 1, len(global_batch_ids), lr, float(loss.numpy()), time.time() - started])
            if (step + 1) % 100 == 0 or step + 1 == len(batches):
                stream.flush()
                print(f"seed={args.seed} step={step + 1}/{len(batches)} batch={len(global_batch_ids)} loss={float(loss.numpy()):.8f}", flush=True)
        saved = manager.save(checkpoint_number=int(global_step.numpy()))
    base_protocol["checkpoint_prefix"] = saved
    if args.smoke:
        weights_changed = any(not np.array_equal(before, after)
                              for before, after in zip(starting_weights, network.get_weights()))
        if not partial_batch_used or not partial_primary_weights_changed or not weights_changed or not finite_losses:
            raise AssertionError("smoke must verify finite loss, 8-row aux path, and primary update")
        base_protocol["smoke_verification"] = {
            "batch_sizes": [len(batch) for batch in batches],
            "auxiliary_final_batch_gradient_applied_to_primary": partial_batch_used,
            "final_partial_update_changed_primary_weights": partial_primary_weights_changed,
            "primary_weights_changed": weights_changed,
            "finite_losses": finite_losses,
        }
    protocol_path.write_text(json.dumps(base_protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "TRAINING_COMPLETED").write_text(f"updates={len(batches)}\ncheckpoint={saved}\n", encoding="utf-8")
    print(f"TRAINING_COMPLETED {saved}", flush=True)


if __name__ == "__main__":
    main()
