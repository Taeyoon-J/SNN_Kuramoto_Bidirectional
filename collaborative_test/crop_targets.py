"""Targets built on the same centre crop Slot Attention sees.

PV2_0090's "resize" arm removes the misalignment but puts the checkpoints
off-distribution, so a drop there is ambiguous. This arm is the clean one: the
prediction is unchanged -- the published crop, as every existing number used -- and
only the TARGET is rebuilt on the same centre 240x240 square. The difference against
the whole-frame targets is the geometry cost alone, with nothing else moved.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/export_home/tkim1/collaborative_test")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")
from evaluation import clevr_mask_patch, evaluate_patch_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecord", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--whole-frame-targets", required=True)
    args = ap.parse_args()

    from training.prepare_clevr_with_masks import iter_records, _parse_example, IMAGE_SHAPE, MAX_ENTITIES
    manifest = json.load(open(args.manifest))
    start, end = manifest["ranges"][args.split]
    end = min(end, start + args.limit)

    frames = []
    for index, record in enumerate(iter_records(args.tfrecord, end)):
        if index < start:
            continue
        feats = _parse_example(record)
        masks = feats["mask"].reshape(MAX_ENTITIES, IMAGE_SHAPE[0], IMAGE_SHAPE[1]) > 127
        frame = torch.zeros(IMAGE_SHAPE[0], IMAGE_SHAPE[1], dtype=torch.int64)
        for entity in range(1, MAX_ENTITIES):
            frame[torch.as_tensor(masks[entity])] = entity
        frames.append(frame)
    pixels = torch.stack(frames)
    height, width = pixels.shape[1], pixels.shape[2]
    side = min(height, width)
    left = (width - side) // 2
    top = (height - side) // 2
    cropped = pixels[:, top:top + side, left:left + side]
    cell = side // args.grid
    assert side % args.grid == 0, "the crop must divide the grid"
    crop_tgt = clevr_mask_patch(cropped, (cell, cell))["patch_labels"]

    whole = torch.load(args.whole_frame_targets, map_location="cpu")["patch_labels"][start:end]

    print("crop %dx%d at (%d,%d), cells of %d px | foreground: crop %.3f, whole %.3f"
          % (side, side, left, top, cell,
             float((crop_tgt != 0).float().mean()), float((whole != 0).float().mean())))
    print("objects per image: crop %.2f, whole %.2f"
          % (np.mean([crop_tgt[i][crop_tgt[i] != 0].unique().numel() for i in range(len(crop_tgt))]),
             np.mean([whole[i][whole[i] != 0].unique().numel() for i in range(len(whole))])))
    print()
    print("%-34s %8s %8s %8s" % ("prediction scored against", "fg_ari", "fg_iou", "obj_iou"))
    for path in args.predictions:
        lab = torch.as_tensor(np.load(path, allow_pickle=True)["patch_labels"]).long()[: len(whole)]
        for name, tgt in (("whole frame (as published)", whole), ("its own crop", crop_tgt)):
            m = evaluate_patch_masks(lab, tgt[: len(lab)])["mean"]
            print("%-34s %8.4f %8.4f %8.4f"
                  % (Path(path).stem[:20] + " / " + name[:12],
                     float(m["fg_ari"]), float(m["foreground_iou"]),
                     float(m["matched_object_iou"])))


if __name__ == "__main__":
    main()
