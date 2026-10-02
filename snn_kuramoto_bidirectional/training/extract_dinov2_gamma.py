"""
Build gamma from DINOv2 patch features, as a baseline the dynamics have to beat.

The features this model is driven by come from an 8-channel CNN trained to
reconstruct CLEVR images. Modern patch-based object discovery is built on
self-supervised features instead -- DINOSAUR, LOST, TokenCut and the deep
spectral methods all partition DINO patch tokens -- so "spectral clustering on
DINOv2 patches" is the baseline that most directly threatens the claim that the
oscillator dynamics are what produce the grouping.

The comparison only means something in pairs: the same features clustered
directly, and the same features run through the core. Reducing to the CNN's
channel count keeps that pair about feature quality rather than width.

DINOv2 uses 14-pixel patches, so an input of grid*14 gives exactly grid x grid
tokens and the grids line up with the rest of the pipeline without resampling.
"""

import argparse
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_images(image_dir, size, max_images):
    paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in {".png", ".jpg"})
    paths = paths[:int(max_images)]
    batch = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
        tensor = torch.frombuffer(image.tobytes(), dtype=torch.uint8).float().div_(255.0)
        batch.append(tensor.view(size, size, 3).permute(2, 0, 1))
    images = torch.stack(batch)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std, [p.name for p in paths]


@torch.no_grad()
def patch_tokens(model, images, grid, device, batch_size=16):
    """[num_images, grid*grid, dim]"""
    out = []
    for start in range(0, images.size(0), batch_size):
        tokens = model.forward_features(images[start:start + batch_size].to(device))
        # drop the class and any register tokens; what remains is the patch grid
        tokens = tokens[:, -(grid * grid):, :]
        out.append(tokens.float().cpu())
    return torch.cat(out)


def pca_reduce(tokens, out_dim, fit_samples=200000):
    """
    Project to out_dim with PCA fitted on the patch vectors themselves.

    Fitted over patches pooled from every image, so the basis describes the
    dataset rather than any one scene.
    """
    flat = tokens.reshape(-1, tokens.size(-1))
    if flat.size(0) > fit_samples:
        index = torch.randperm(flat.size(0))[:fit_samples]
        fit = flat[index]
    else:
        fit = flat
    mean = fit.mean(dim=0, keepdim=True)
    _, _, basis = torch.pca_lowrank(fit - mean, q=min(out_dim + 8, fit.size(1)))
    basis = basis[:, :out_dim]
    return (flat - mean) @ basis, mean, basis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--out-dim", type=int, default=8,
                        help="channel count; match the CNN's so the pair compares features, not width")
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--clip", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    size = args.grid * 14
    images, names = load_images(args.image_dir, size, args.max_images)
    print("images %s at %dx%d -> %dx%d tokens"
          % (tuple(images.shape), size, size, args.grid, args.grid), flush=True)

    model = timm.create_model(args.model, pretrained=True, num_classes=0, img_size=size)
    model.eval().to(args.device)
    tokens = patch_tokens(model, images, args.grid, args.device)
    print("patch tokens %s" % (tuple(tokens.shape),), flush=True)

    reduced, _, _ = pca_reduce(tokens, args.out_dim)
    reduced = reduced.view(tokens.size(0), args.grid * args.grid, args.out_dim)

    # same normalisation the CNN gamma pipeline applies, so the drive sees a
    # comparable scale and gamma_phase_mode behaves the same way
    reduced = (reduced - reduced.mean()) / reduced.std().clamp_min(1e-8)
    reduced = reduced.clamp(-args.clip, args.clip)

    gamma = reduced.permute(0, 2, 1).contiguous()      # [images, out_dim, patches]
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(gamma, args.save_path)
    print("saved %s to %s" % (tuple(gamma.shape), args.save_path))


if __name__ == "__main__":
    main()
