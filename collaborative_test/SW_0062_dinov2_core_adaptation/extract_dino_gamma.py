#!/usr/bin/env python3
"""Extract label-free DINOv2 patch tokens and fit training-only PCA gamma."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import timm
import torch
import torch.nn.functional as F

MODEL = "vit_small_patch14_dinov2.lvd142m"
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@torch.inference_mode()
def extract(model, dataset, ids, batch_size, device):
    chunks = []
    for start in range(0, len(ids), batch_size):
        chosen = ids[start:start + batch_size]
        images = torch.from_numpy(dataset["image"][chosen]).permute(0, 3, 1, 2).float().div_(255.0)
        images = F.interpolate(images, (224, 224), mode="bicubic", align_corners=False)
        images = (images - IMAGENET_MEAN) / IMAGENET_STD
        tokens = model.forward_features(images.to(device))
        tokens = tokens[:, model.num_prefix_tokens:]
        if tuple(tokens.shape[1:]) != (256, 384):
            raise RuntimeError(f"unexpected DINO token shape {tuple(tokens.shape)}")
        chunks.append(tokens.cpu().to(torch.float16))
    return torch.cat(chunks)


def project(tokens, mean, basis, batch_size=64):
    rows = []
    for start in range(0, len(tokens), batch_size):
        x = tokens[start:start + batch_size].float()
        reduced = torch.matmul(x - mean, basis)
        reduced = F.normalize(reduced, dim=-1)
        rows.append(reduced.transpose(1, 2).contiguous())
    result = torch.cat(rows)
    if not torch.isfinite(result).all():
        raise RuntimeError("non-finite projected DINO gamma")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    outputs = {
        "train": args.output_dir / "dino_gamma_train_0_999.pt",
        "validation": args.output_dir / "dino_gamma_validation_1320_1639.pt",
        "pca": args.output_dir / "dino_pca_8d.pt",
        "manifest": args.output_dir / "manifest.json",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("refusing to overwrite DINO feature assets")
    torch.set_num_threads(args.threads)
    torch.manual_seed(62)
    device = torch.device(args.device)
    model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=224).to(device).eval()
    train_ids = list(range(0, 1000))
    validation_ids = list(range(1320, 1640))
    with h5py.File(args.hdf5, "r") as dataset:
        train_tokens = extract(model, dataset, train_ids, args.batch_size, device)
        validation_tokens = extract(model, dataset, validation_ids, args.batch_size, device)
    sample = train_tokens[:, ::4, :].reshape(-1, 384).float()
    mean = sample.mean(dim=0)
    _, singular_values, basis = torch.pca_lowrank(sample - mean, q=8, center=False, niter=5)
    train_gamma = project(train_tokens, mean, basis)
    validation_gamma = project(validation_tokens, mean, basis)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_gamma, outputs["train"])
    torch.save(validation_gamma, outputs["validation"])
    torch.save({"mean": mean, "basis": basis, "singular_values": singular_values}, outputs["pca"])
    manifest = {
        "experiment": "SW0062 frozen DINOv2 patch features",
        "model": MODEL,
        "timm_version": timm.__version__,
        "input_size": [224, 224],
        "patch_grid": [16, 16],
        "token_dim": 384,
        "pca_dim": 8,
        "pca_fit_ids": [0, 999],
        "train_ids": [0, 999],
        "validation_ids": [1320, 1639],
        "uses_masks_counts_or_labels": False,
        "shapes": {"train": list(train_gamma.shape), "validation": list(validation_gamma.shape)},
        "sha256": {"train": sha256(outputs["train"]), "validation": sha256(outputs["validation"]), "pca": sha256(outputs["pca"])},
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
