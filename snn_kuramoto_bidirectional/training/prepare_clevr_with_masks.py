"""
Convert DeepMind's clevr_with_masks into images plus patch-level object labels.

CLEVR v1.0, which this project has been using, ships images and a scene file but
no segmentation. The masks used until now were reconstructed by matching each
object's declared colour against the pixels near its stated centre, and that
recovers only 77% of the objects in a scene: anything occluded, in shadow, or
made of the reflective material loses too many pixels to the coverage test.
Objects the model does find are then scored as errors.

clevr_with_masks is the version object-discovery papers evaluate on, so using it
also makes the numbers comparable to published foreground ARI. It is a different
render -- 320x240 against CLEVR v1.0's 480x320 -- so the images come from here
too, and gamma has to be regenerated.

No TensorFlow on the cluster, so the reader is written directly against the two
formats involved. Both are simple: a TFRecord is length-prefixed frames with
CRCs, and tf.train.Example is a small protobuf.
"""

import argparse
import gzip
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_SHAPE = (240, 320, 3)
MAX_ENTITIES = 11
MIN_COVER = 0.30


def _read_varint(buf, pos):
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _parse_bytes_list(payload):
    """
    Concatenate a BytesList's entries.

    Every entry in this dataset holds exactly one byte, which encodes as the
    fixed three-byte pattern 0x0A 0x01 <value>. An image is 230400 entries and a
    mask stack is 844800, so walking them one varint at a time costs about a
    million steps per record. When the pattern holds the whole list is one numpy
    stride instead.
    """
    raw = np.frombuffer(payload, dtype=np.uint8)
    if raw.size % 3 == 0 and raw.size:
        head = raw[0::3]
        length = raw[1::3]
        if (head == 0x0A).all() and (length == 0x01).all():
            return raw[2::3].copy()

    out, pos, end = bytearray(), 0, len(payload)
    while pos < end:
        key, pos = _read_varint(payload, pos)
        if key >> 3 != 1 or key & 7 != 2:
            raise ValueError("unexpected field in BytesList")
        size, pos = _read_varint(payload, pos)
        out += payload[pos:pos + size]
        pos += size
    return np.frombuffer(bytes(out), dtype=np.uint8).copy()


def _parse_example(record):
    """Field name -> numpy array, for the features this converter needs."""
    features = {}
    pos, end = 0, len(record)
    while pos < end:                                    # Example { Features = 1 }
        key, pos = _read_varint(record, pos)
        size, pos = _read_varint(record, pos)
        if key >> 3 != 1:
            pos += size
            continue
        block, pos = record[pos:pos + size], pos + size

        inner = 0
        while inner < len(block):                       # Features { map entry = 1 }
            key2, inner = _read_varint(block, inner)
            size2, inner = _read_varint(block, inner)
            entry, inner = block[inner:inner + size2], inner + size2
            if key2 >> 3 != 1:
                continue

            name, value, cur = None, None, 0
            while cur < len(entry):
                key3, cur = _read_varint(entry, cur)
                size3, cur = _read_varint(entry, cur)
                chunk, cur = entry[cur:cur + size3], cur + size3
                if key3 >> 3 == 1:
                    name = chunk.decode()
                elif key3 >> 3 == 2:                    # Feature { oneof kind }
                    kind, at = _read_varint(chunk, 0)
                    size4, at = _read_varint(chunk, at)
                    # Only bytes_list is read. The float and int fields carry
                    # object attributes this converter does not use, and each
                    # would need its own packed/unpacked handling.
                    if kind >> 3 == 1:
                        value = _parse_bytes_list(chunk[at:at + size4])
            if name is not None and value is not None:
                features[name] = value
    return features


def iter_records(path, limit=None):
    """Yield raw record payloads from a GZIP-compressed TFRecord file."""
    count = 0
    with gzip.open(path, "rb") as handle:
        while limit is None or count < limit:
            header = handle.read(8)
            if len(header) < 8:
                return
            length = struct.unpack("<Q", header)[0]
            handle.read(4)                              # CRC of the length
            payload = handle.read(length)
            handle.read(4)                              # CRC of the payload
            if len(payload) < length:
                return
            yield payload
            count += 1


def masks_to_patch_labels(masks, grid):
    """
    Object id per patch, -1 for background. [grid*grid]

    Same convention as the colour-derived masks it replaces, so everything
    downstream is unchanged: a patch belongs to whichever object covers at
    least MIN_COVER of it and covers it most. Entity 0 is background.
    """
    labels = torch.full((grid * grid,), -1, dtype=torch.long)
    best = torch.zeros(grid * grid)
    for entity in range(1, masks.shape[0]):
        area = torch.as_tensor(masks[entity], dtype=torch.float32)
        if area.sum() < 1:
            continue
        cover = F.adaptive_avg_pool2d(area.view(1, 1, *area.shape), (grid, grid)).flatten()
        take = (cover >= MIN_COVER) & (cover > best)
        labels[take] = entity - 1
        best[take] = cover[take]
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord", required=True)
    parser.add_argument("--image-dir", required=True, help="where the PNGs are written")
    parser.add_argument("--masks-out", required=True, help=".pt with names and patch labels")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--grid", type=int, nargs="+", default=[16, 32])
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    names, per_grid = [], {g: [] for g in args.grid}
    counts = []

    for index, record in enumerate(iter_records(args.tfrecord, args.num_images)):
        features = _parse_example(record)
        image = features["image"].reshape(IMAGE_SHAPE)
        masks = features["mask"].reshape(MAX_ENTITIES, IMAGE_SHAPE[0], IMAGE_SHAPE[1])
        masks = (masks > 127).astype(np.uint8)

        name = "CLEVR_masks_%06d.png" % index
        Image.fromarray(image).save(image_dir / name)
        names.append(name)
        for g in args.grid:
            per_grid[g].append(masks_to_patch_labels(masks, g))
        counts.append(int(sum(masks[e].sum() > 0 for e in range(1, MAX_ENTITIES))))
        if (index + 1) % 200 == 0:
            print("  %d images" % (index + 1), flush=True)

    blob = {"names": names,
            **{"labels_grid%d" % g: torch.stack(per_grid[g]) for g in args.grid}}
    torch.save(blob, args.masks_out)
    print("wrote %d images to %s" % (len(names), image_dir))
    print("wrote labels for grids %s to %s" % (args.grid, args.masks_out))
    print("objects per image: mean %.2f, max %d" % (sum(counts) / len(counts), max(counts)))
    for g in args.grid:
        stack = torch.stack(per_grid[g])
        recovered = sum(int(stack[i].max()) + 1 for i in range(len(names)) if stack[i].max() >= 0)
        print("grid %2d: foreground %.1f%% of patches, %.2f objects per image visible at this grid"
              % (g, 100 * float((stack >= 0).float().mean()), recovered / len(names)))


if __name__ == "__main__":
    main()
