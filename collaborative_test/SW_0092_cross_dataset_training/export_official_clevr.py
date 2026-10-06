"""Export the exact released Slot Attention CLEVR6 training stream to HDF5."""
import argparse
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

p = argparse.ArgumentParser()
p.add_argument("--tfds-dir", default="/Data0/kevinswk/datasets/tfds")
p.add_argument("--output", required=True)
p.add_argument("--manifest", required=True)
p.add_argument("--batch-size", type=int, default=128)
a = p.parse_args()
output, manifest = Path(a.output), Path(a.manifest)
if output.exists() or manifest.exists():
    raise FileExistsError("refusing to overwrite export or manifest")
output.parent.mkdir(parents=True, exist_ok=True)

ds = tfds.load("clevr:3.1.0", split="train", shuffle_files=False, data_dir=a.tfds_dir)
ds = ds.enumerate().skip(512)
ds = ds.filter(lambda index, row: tf.shape(row["objects"]["3d_coords"])[0] <= 6)

def preprocess(index, row):
    image = tf.cast(row["image"][29:221, 64:256, :], tf.float32)
    image = tf.image.resize(image, (128, 128), method=tf.image.ResizeMethod.BILINEAR)
    image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
    return index, tf.shape(row["objects"]["3d_coords"])[0], image

ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(a.batch_size)
with h5py.File(output, "x") as h5:
    images = h5.create_dataset("image", shape=(0, 128, 128, 3), maxshape=(None, 128, 128, 3),
                               dtype="float16", chunks=(16, 128, 128, 3))
    source_ids = h5.create_dataset("tfds_train_index", shape=(0,), maxshape=(None,), dtype="int64")
    counts = h5.create_dataset("num_actual_objects", shape=(0,), maxshape=(None,), dtype="int8")
    total = 0
    for batch_index, batch_count, batch_image in tfds.as_numpy(ds):
        n = len(batch_index)
        for dataset in (images, source_ids, counts): dataset.resize(total + n, axis=0)
        images[total:total+n] = batch_image.astype(np.float16)
        source_ids[total:total+n] = batch_index
        counts[total:total+n] = batch_count
        total += n
        if total % 4096 < n: print(f"EXPORTED {total}", flush=True)
record = {
    "source": "TFDS clevr:3.1.0 train",
    "tfds_data_dir": a.tfds_dir,
    "official_order_operations": ["enumerate train", "skip first 512", "filter object_count <= 6"],
    "preprocessing": "crop image[29:221,64:256], bilinear resize 128x128, float16 [0,1]",
    "images": total,
    "output": str(output),
    "ground_truth_masks_available": False,
}
manifest.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2), flush=True)

