"""Run the trained SW0240 model on 128x128 RGB images."""
import argparse
from pathlib import Path
import numpy as np
import torch
from snn_kuramoto_bidirectional import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='uint8 [B,128,128,3] .npy or a single 128x128 RGB image')
    parser.add_argument('--output', type=Path, required=True, help='output .npz: labels [B,64,64], 0=background')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed-base', type=int, default=193000)
    args = parser.parse_args()
    if args.output.suffix.lower() != '.npz':
        raise ValueError('--output must end in .npz')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.input.suffix.lower() == '.npy':
        rgb = np.load(args.input, allow_pickle=False)
    else:
        from PIL import Image
        rgb = np.asarray(Image.open(args.input).convert('RGB'))[None]
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[1:] != (128,128,3) or not len(rgb):
        raise ValueError('input must be nonempty uint8 RGB [B,128,128,3]; images are not resized automatically')
    model = load_model(args.checkpoint, device=args.device)
    labels = model.predict(rgb, inference_seeds=range(args.seed_base,args.seed_base+len(rgb)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, labels=labels)
    print(f'Saved {len(labels)} native64 masks to {args.output}')


if __name__ == '__main__':
    main()
