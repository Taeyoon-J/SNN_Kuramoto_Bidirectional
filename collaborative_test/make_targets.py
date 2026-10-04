"""Build the evaluation targets the contract specifies.

The patch labels used until now came from a local rule -- an object claimed a
patch if it covered 30% of it, background was -1 -- which is not what the
contract fixes. `clevr_mask_patch` takes each patch's most frequent instance ID
with ties going to the smallest, so background wins a tie, and background is 0.

Reads the pixel masks straight from the tfrecord, so nothing depends on the
earlier conversion.
"""
import argparse, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "snn_kuramoto_bidirectional"))
from evaluation import clevr_mask_patch
from training.prepare_clevr_with_masks import iter_records, _parse_example, IMAGE_SHAPE, MAX_ENTITIES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecord", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch-size", type=int, nargs=2, default=[15, 20],
                    help="240/15 = 16 and 320/20 = 16, so the grid is 16x16")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    limit = max(end for _, end in manifest["ranges"].values())
    height, width = IMAGE_SHAPE[0], IMAGE_SHAPE[1]

    labels, names = [], []
    for index, record in enumerate(iter_records(args.tfrecord, limit)):
        feats = _parse_example(record)
        masks = feats["mask"].reshape(MAX_ENTITIES, height, width) > 127
        # instance ids 0..10 with 0 as background, exactly the dataset's own
        pixel = torch.zeros(height, width, dtype=torch.int64)
        for entity in range(1, MAX_ENTITIES):
            pixel[torch.as_tensor(masks[entity])] = entity
        labels.append(pixel)
        names.append("CLEVR_masks_%06d.png" % index)
        if (index + 1) % 1000 == 0:
            print("  %d images" % (index + 1), flush=True)

    patched = clevr_mask_patch(torch.stack(labels), tuple(args.patch_size))
    blob = {
        "names": names,
        "patch_labels": patched["patch_labels"],
        "patch_purity": patched["patch_purity"],
        "patch_ties": patched["patch_ties"],
        "patch_size": list(args.patch_size),
        "contract_version": 1,
        "manifest_version": manifest["version"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)

    lab = patched["patch_labels"]
    fg = (lab != 0)
    objects = torch.stack([(lab[i][fg[i]].unique().numel()) * torch.ones(1)
                           for i in range(lab.size(0))]).mean()
    print("wrote %s  labels %s" % (args.out, tuple(lab.shape)))
    print("foreground %.1f%% of patches | %.2f objects per image | ties on %.2f%% of patches"
          % (100 * fg.float().mean(), float(objects),
             100 * patched["patch_ties"].float().mean()))
    print("mean purity of the chosen id: %.3f" % float(patched["patch_purity"].mean()))


if __name__ == "__main__":
    main()
