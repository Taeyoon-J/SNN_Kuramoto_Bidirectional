"""Check whether indexed CLEVR PNGs match the HDF5 images used as targets."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def rgb32(pixels):
    image = Image.fromarray(pixels.astype(np.uint8)).convert("RGB")
    return np.asarray(image.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    indices = (0, 1, 2, 5, 9, 20, 50, 99, 100, 200, 500, 900)
    with h5py.File(args.dataset_path, "r") as dataset:
        images = dataset["image"][:1000]
    if images.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB HDF5 images, got {images.shape}")
    if images.dtype != np.uint8:
        images = np.clip(images * (255 if images.max() <= 1 else 1), 0, 255).astype(np.uint8)
    reference = np.stack([rgb32(image) for image in images])
    rows = []
    for index in indices:
        png_path = Path(args.png_dir) / f"CLEVR_train_{index:06d}.png"
        source = Image.open(png_path).convert("RGB")
        candidate = np.asarray(source.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        distances = np.square(reference - candidate).mean(axis=(1, 2, 3))
        nearest = int(distances.argmin())
        rows.append({"png_index": index, "nearest_hdf5_index": nearest,
                     "same_index_mse": float(distances[index]),
                     "nearest_mse": float(distances[nearest]),
                     "same_index_rank": int(np.sum(distances < distances[index])) + 1})
    result = {"png_dir": args.png_dir, "dataset_path": args.dataset_path,
              "hdf5_shape": list(images.shape), "samples": rows,
              "same_index_nearest_count": sum(row["png_index"] == row["nearest_hdf5_index"] for row in rows)}
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
