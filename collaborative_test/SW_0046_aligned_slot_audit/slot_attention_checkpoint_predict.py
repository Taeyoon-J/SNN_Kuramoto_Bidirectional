"""Run official Slot Attention inference on a parameterized HDF5 slice."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "4"

import argparse
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

from protocol import perimeter_background, remap_foreground, validate_slice


def load_model_module(model_py):
    model_py = Path(model_py).resolve()
    if not model_py.is_file():
        raise FileNotFoundError(f"Slot Attention model.py not found: {model_py}")
    model_directory = str(model_py.parent)
    if model_directory not in sys.path:
        sys.path.insert(0, model_directory)
    spec = importlib.util.spec_from_file_location("slot_attention_official_model", model_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model module from {model_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_checkpoint(tf, checkpoint_dir, checkpoint_prefix=None):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if checkpoint_prefix:
        prefix = Path(checkpoint_prefix)
        if not prefix.is_absolute():
            prefix = checkpoint_dir / prefix
        checkpoint_path = str(prefix.resolve())
    else:
        checkpoint_path = tf.train.latest_checkpoint(str(checkpoint_dir))
    if not checkpoint_path or not Path(checkpoint_path + ".index").is_file():
        raise FileNotFoundError(
            f"TensorFlow checkpoint index not found under {checkpoint_dir}: {checkpoint_path}")
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-prefix", default=None,
                        help="Optional prefix such as ckpt-500; otherwise use latest_checkpoint().")
    parser.add_argument("--model-py", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.random.set_seed(0)
        model_py = Path(args.model_py).resolve()
        model = load_model_module(model_py)
        checkpoint_path = resolve_checkpoint(tf, args.checkpoint_dir, args.checkpoint_prefix)
        network = model.build_model((128, 128), 1, 11, 3, model_type="object_discovery")
        checkpoint = tf.train.Checkpoint(network=network)
        status = checkpoint.restore(checkpoint_path)
        status.assert_existing_objects_matched()
        status.expect_partial()  # Training optimizer/global_step are not inference variables.
        print("ALL_MODEL_VARIABLES_RESTORED", len(network.weights), flush=True)

        with h5py.File(args.dataset, "r") as source:
            image_dataset = source["image"]
            start, end = validate_slice(args.start, args.count, image_dataset.shape[0])
            images = image_dataset[start:end]
        predictions, background_slots, reconstruction_mse = [], [], []
        for offset, rgb in enumerate(images):
            # Faithful official protocol: full 128x128 HDF5 image, no extra crop.
            x = rgb.astype(np.float32)[None] / 127.5 - 1.0
            reconstruction, _, masks, _ = network(x, training=False)
            slot_ids = masks.numpy()[0, ..., 0].argmax(axis=0)
            bg = perimeter_background(slot_ids, num_slots=11)
            labels = remap_foreground(slot_ids, bg, num_slots=11)
            predictions.append(labels)
            background_slots.append(bg)
            reconstruction_mse.append(float(np.mean((reconstruction.numpy() - x) ** 2)))
            if (offset + 1) % 20 == 0:
                print(f"INFERENCE {offset + 1}/{args.count}", flush=True)

        image_ids = np.arange(start, end, dtype=np.int64)
        np.savez_compressed(
            output_dir / "predictions.npz",
            image_ids=image_ids,
            labels=np.stack(predictions),
            background_slots=np.asarray(background_slots, dtype=np.int64),
            reconstruction_mse=np.asarray(reconstruction_mse, dtype=np.float32),
        )
        protocol = {
            "dataset": str(Path(args.dataset).resolve()),
            "image_ids": [start, end - 1],
            "count": args.count,
            "seed": 0,
            "resolution": [128, 128],
            "batch_size": 1,
            "num_slots": 11,
            "iterations": 3,
            "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
            "checkpoint_prefix": checkpoint_path,
            "checkpoint_source": "gs://gresearch/slot-attention/object-discovery/ckpt-500",
            "model_py": str(model_py),
            "model_sha256": hashlib.sha256(model_py.read_bytes()).hexdigest(),
            "tensorflow_version": tf.__version__,
            "preprocessing": "full HDF5 RGB image, float32 /127.5 - 1; no crop",
            "background_rule": "most hard-assigned one-pixel perimeter pixels; smallest slot ID wins ties",
            "label_rule": "slot ID + 1; selected background slot set to 0",
            "ground_truth_used_for_prediction": False,
        }
        (output_dir / "protocol.json").write_text(
            json.dumps(protocol, indent=2, allow_nan=False), encoding="utf-8")
        (output_dir / "INFERENCE_COMPLETED").write_text(
            f"{args.count} images\n", encoding="utf-8")
        print("INFERENCE_COMPLETED", flush=True)
    except Exception:
        (output_dir / "FAILED").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
