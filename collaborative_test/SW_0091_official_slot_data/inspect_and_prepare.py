"""Inspect or download the exact TFDS CLEVR release used by released Slot Attention."""
import argparse
import json
from pathlib import Path

import tensorflow_datasets as tfds

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="/Data0/kevinswk/datasets/tfds")
p.add_argument("--download", action="store_true")
a = p.parse_args()
builder = tfds.builder("clevr:3.1.0", data_dir=a.data_dir)
info = {
    "tfds_version": tfds.__version__,
    "dataset": builder.info.full_name,
    "data_dir": str(builder.data_dir),
    "download_size": str(builder.info.download_size),
    "dataset_size": str(builder.info.dataset_size),
    "splits": {name: split.num_examples for name, split in builder.info.splits.items()},
    "prepared_before": builder.data_path.exists(),
}
print(json.dumps(info, indent=2), flush=True)
if a.download:
    builder.download_and_prepare()
    info["prepared_after"] = builder.data_path.exists()
    info["prepared_path"] = str(builder.data_path)
    manifest = Path(a.data_dir) / "SW0091_official_clevr_manifest.json"
    manifest.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)

