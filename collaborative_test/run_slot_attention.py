"""Run the published Slot Attention checkpoint on our test split.

The reference this project is measured against. Per the contract its masks go
through the same patch conversion and the same evaluation functions as ours, on
the same images in the same order.

Two things about it are stated rather than hidden. It is **one published
checkpoint**, so its column is a single-checkpoint reference and never a 3-seed
training mean. And it was trained on the original CLEVR render, while the split
here comes from clevr_with_masks, which is a different render with different
object statistics -- that disadvantages it, and the authors say as much in their
own README.

Its preprocessing is theirs: centre crop to 192x192 of the original frame, resize
to 128x128, scale to [-1, 1]. The slot masks come back at 128x128 and are mapped
to the 16x16 grid by majority vote per patch, the same rule clevr_mask_patch
applies to the ground truth.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--model-dir", required=True, help="where model.py was downloaded")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-slots", type=int, default=7)
    ap.add_argument("--num-iterations", type=int, default=3)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--out", required=True, help="npz of patch labels")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.model_dir).parent))
    import tensorflow as tf
    from PIL import Image
    sys.path.insert(0, args.model_dir)
    import model as sa_model

    manifest = json.load(open(args.manifest))
    names = manifest["splits"][args.split]
    if args.limit:
        names = names[: args.limit]

    net = sa_model.build_model(
        resolution=(128, 128), batch_size=1,
        num_slots=args.num_slots, num_iterations=args.num_iterations,
        model_type="object_discovery",
    )
    ckpt = tf.train.Checkpoint(network=net)
    manager = tf.train.CheckpointManager(ckpt, args.checkpoint_dir, max_to_keep=1)
    status = ckpt.restore(manager.latest_checkpoint)
    print("restored from", manager.latest_checkpoint, flush=True)

    patch_h = 128 // args.grid
    labels = np.zeros((len(names), args.grid, args.grid), dtype=np.int64)
    for i, name in enumerate(names):
        image = Image.open(Path(args.image_dir) / name).convert("RGB")
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        image = image.crop((left, top, left + side, top + side)).resize((128, 128), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0 * 2.0 - 1.0
        _, _, masks, _ = net(array[None])
        # masks: [1, num_slots, 128, 128, 1] -> hard assignment per pixel
        assignment = np.argmax(np.asarray(masks)[0, ..., 0], axis=0)
        # majority vote per patch, the same rule the ground truth uses
        tiles = assignment.reshape(args.grid, patch_h, args.grid, patch_h)
        tiles = tiles.transpose(0, 2, 1, 3).reshape(args.grid, args.grid, -1)
        for r in range(args.grid):
            for c in range(args.grid):
                labels[i, r, c] = np.bincount(tiles[r, c], minlength=args.num_slots).argmax()
        if (i + 1) % 100 == 0:
            print("  %d images" % (i + 1), flush=True)

    # Slot ids are arbitrary; the background slot is whichever covers the most
    # patches overall, and is remapped to 0 so the contract's background id holds.
    background = np.bincount(labels.reshape(-1), minlength=args.num_slots).argmax()
    remapped = np.where(labels == background, 0, labels + 1)
    remapped = np.where(labels == background, 0, remapped)
    np.savez(args.out, patch_labels=remapped, names=np.array(names),
             background_slot=background, num_slots=args.num_slots)
    print("wrote %s  %s  background slot %d, %.1f%% of patches"
          % (args.out, remapped.shape, background,
             100 * (remapped == 0).mean()))


if __name__ == "__main__":
    main()
