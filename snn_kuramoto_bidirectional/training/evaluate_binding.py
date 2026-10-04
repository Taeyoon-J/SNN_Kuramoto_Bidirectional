"""
Score a trained core's grouping against CLEVR object masks.

This lived only in ad-hoc scripts under /tmp on the cluster and was rewritten
from scratch every session, which is how the readout comparison ended up being
run against the wrong baseline more than once. It is a file now.

Every readout the project has is scored side by side, at the oracle cluster
count and at fixed counts, because the choice of k turned out to matter more
than the choice of readout: on the phase route a fixed k=3 beats the oracle.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# CLEVR names its object colours, and the renderer paints them from a fixed
# palette, so an object mask can be recovered from the image without any
# segmentation ground truth. This captures about 78% of the scene objects:
# objects that are occluded, in shadow, or made of the reflective material lose
# too many pixels to the coverage test.
PALETTE = {
    "gray": (87, 87, 87), "red": (173, 35, 35), "blue": (42, 75, 215),
    "green": (29, 105, 20), "brown": (129, 74, 25), "purple": (129, 38, 192),
    "cyan": (41, 208, 208), "yellow": (255, 238, 51),
}
MIN_COVER = 0.30


def build_masks(names, objects_by_name, image_dir, grid):
    """
    Object id per patch, -1 for background. [len(names), grid*grid]

    Ported here from a script that lived outside the repository with the grid
    size fixed at 16, which is what made a 32x32 comparison impossible to run.
    """
    image_dir = Path(image_dir)
    labels = torch.full((len(names), grid * grid), -1, dtype=torch.long)
    for bi, name in enumerate(names):
        image = Image.open(image_dir / name).convert("RGB")
        width, height = image.size
        pixels = torch.as_tensor(list(image.getdata()), dtype=torch.float32).view(height, width, 3)
        yy, xx = torch.meshgrid(torch.arange(height).float(),
                                torch.arange(width).float(), indexing="ij")
        best = torch.zeros(grid * grid)
        for oi, obj in enumerate(objects_by_name[name]):
            colour = torch.tensor(PALETTE[obj["color"]], dtype=torch.float32)
            cx, cy = obj["pixel_coords"][0], obj["pixel_coords"][1]
            near = ((xx - cx) ** 2 + (yy - cy) ** 2).sqrt() < 60.0
            mask = ((pixels - colour).norm(dim=-1) < 60.0) & near
            if mask.sum() < 20:
                continue
            cover = F.adaptive_avg_pool2d(mask.float().view(1, 1, height, width),
                                          (grid, grid)).flatten()
            take = (cover >= MIN_COVER) & (cover > best)
            labels[bi][take] = oi
            best[take] = cover[take]
    return labels

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
    from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
    from snn_kuramoto_bidirectional.loss_function import (
        phase_locking_value,
        signal_synchrony,
    )
except ModuleNotFoundError:
    from hyperparameter import S2NetHyperparameters
    from s2net_cls import S2NetCore
    from loss_function import phase_locking_value, signal_synchrony


def adjusted_rand_index(a, b):
    """ARI between two label vectors, without sklearn."""
    a = torch.unique(a, return_inverse=True)[1]
    b = torch.unique(b, return_inverse=True)[1]
    table = torch.zeros(int(a.max()) + 1, int(b.max()) + 1)
    table.index_put_((a, b), torch.ones(a.numel()), accumulate=True)
    comb = lambda x: (x * (x - 1) / 2).sum()
    index = comb(table)
    expected = comb(table.sum(1)) * comb(table.sum(0)) / max(comb(torch.tensor([a.numel()])), 1e-12)
    maximum = (comb(table.sum(1)) + comb(table.sum(0))) / 2
    return float((index - expected) / (maximum - expected).clamp_min(1e-12))


def kmeans(x, k, iters=30):
    """Spherical k-means with farthest-point seeds, on the rows of x."""
    z = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    centres = z[(z @ z.mean(0, keepdim=True).t()).squeeze(1).argmin()].unsqueeze(0)
    while centres.size(0) < k:
        centres = torch.cat([centres, z[(z @ centres.t()).max(1).values.argmin()].unsqueeze(0)])
    for _ in range(iters):
        assign = (z @ centres.t()).argmax(1)
        for j in range(k):
            members = assign == j
            if members.any():
                centres[j] = torch.nn.functional.normalize(z[members].mean(0), dim=0)
    return (z @ centres.t()).argmax(1)


def spectral_cluster(affinity, k, jitter=1e-6):
    """Normalized-cut spectral clustering. The jitter is not cosmetic: without it
    eigh fails to converge on the degenerate affinities this model produces."""
    affinity = affinity + jitter * torch.eye(affinity.size(0), dtype=affinity.dtype)
    degree = affinity.sum(1).clamp_min(1e-8).rsqrt()
    laplacian = degree.unsqueeze(1) * affinity * degree.unsqueeze(0)
    try:
        _, vectors = torch.linalg.eigh(laplacian)
    except Exception:
        _, vectors = torch.linalg.eigh(laplacian.double())
        vectors = vectors.float()
    embedding = vectors[:, -k:]
    return kmeans(embedding, k)


def foreground_iou(pred, truth):
    """Mean over true objects of the best IoU against any predicted group."""
    background = int(torch.bincount(truth).argmax())
    scores = []
    for gid in truth.unique().tolist():
        if gid == background:
            continue
        gm = truth == gid
        scores.append(max(
            float((gm & (pred == pid)).sum()) / max(float((gm | (pred == pid)).sum()), 1)
            for pid in pred.unique().tolist()
        ))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-seq-path", required=True)
    parser.add_argument("--masks", help="blob supplying the image name list")
    parser.add_argument("--image-dir", help="CLEVR train images")
    parser.add_argument(
        "--patch-labels",
        help=(
            "Blob from prepare_clevr_with_masks.py holding labels_grid<N>. These are "
            "the dataset's own segmentation, so they replace both --masks and the "
            "colour reconstruction, which recovered 77%% of objects and 54%% of the "
            "foreground area."
        ),
    )
    parser.add_argument("--scenes", help="CLEVR_train_scenes.json")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--skip", type=int, default=200, help="images to skip, to score held-out ones")
    parser.add_argument("--num-regions", type=int, default=256)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--num-time-steps", type=int, default=256)
    parser.add_argument("--settle", type=int, default=64)
    parser.add_argument(
        "--graph-top-k",
        type=int,
        default=32,
        help=(
            "Must match training. top_k selects edges rather than parameterising "
            "them, so a mismatch loads without error and silently evaluates a "
            "different coupling graph than the one that was trained."
        ),
    )
    parser.add_argument(
        "--graph-spatial-decay",
        type=float,
        default=0.55,
        help="Must match the spatial decay used during training.",
    )
    parser.add_argument("--freq-gain", type=float, default=2.0)
    parser.add_argument("--osc-dim", type=int, default=4, help="must match training")
    parser.add_argument("--spike-per-component", action="store_true", help="must match training")
    parser.add_argument(
        "--gate-mode",
        choices=["sigmoid", "raw", "phase_mean"],
        default="raw",
        help=(
            "Must match training. phase_mean hands the dendrite one reduced "
            "oscillation per unit and the others hand it osc_dim, so a mismatch is "
            "a state_dict shape error rather than a silently wrong score."
        ),
    )
    parser.add_argument("--fixed-k", type=int, nargs="*", default=[3, 4, 6, 8])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.patch_labels:
        blob = torch.load(args.patch_labels)
        names = blob["names"][args.skip:args.skip + args.num_images]
        key = "labels_grid%d" % args.grid
        if key not in blob:
            raise ValueError("%s has no %s; available: %s"
                             % (args.patch_labels, key, sorted(blob)))
        labels = blob[key][args.skip:args.skip + args.num_images]
        objects_by_name = None
    else:
        if not (args.masks and args.image_dir and args.scenes):
            raise ValueError("pass --patch-labels, or all of --masks, --image-dir, --scenes.")
        masks_blob = torch.load(args.masks)
        names = masks_blob["names"][args.skip:args.skip + args.num_images]
        scenes = json.load(open(args.scenes))["scenes"]
        objects_by_name = {s["image_filename"]: s["objects"]
                           for s in scenes if s["image_filename"] in set(names)}
        labels = build_masks(names, objects_by_name, args.image_dir, args.grid)

    gamma = torch.load(args.gamma_seq_path, map_location="cpu").float()
    gamma = gamma[args.skip:args.skip + args.num_images].to(args.device)

    hp = S2NetHyperparameters(
        num_feature_maps=gamma.size(1), num_regions=args.num_regions, sc=None,
        gamma_drive_mode="static", num_time_steps=args.num_time_steps,
        theta_init="gamma", gamma_phase_mode="standardize_tanh", osc_dim=args.osc_dim,
        freq_gain=args.freq_gain,
        graph_mode="learned", graph_top_k=args.graph_top_k, k=256.,
        graph_spatial_decay=args.graph_spatial_decay,
        low_n=-4., high_n=0., membrane_vth=0.06, membrane_low_m=-4., membrane_high_m=0.,
        gate_mode=args.gate_mode, spike_classify_method="spatial_components",
        spike_spatial_grid_size=(args.grid, args.grid),
        spike_per_component=args.spike_per_component,
    ).validate()
    core = S2NetCore(hp, device=args.device).to(args.device)
    core.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    core.eval()

    plv, plv_parts, spike_sync, trains = [], [], [], []
    spike_parts = []
    with torch.no_grad():
        for start in range(0, gamma.size(0), 25):
            _, _, membrane, theta = core(gamma[start:start + 25],
                                         return_core_out=True, return_theta=True)
            spikes = (membrane > torch.quantile(
                membrane[:, :, args.settle:], 0.50, dim=2, keepdim=True)).float()
            plv.append(phase_locking_value(theta, settle=args.settle).cpu())
            plv_parts.append(
                phase_locking_value(theta, settle=args.settle, combine="product").cpu()
            )
            spike_sync.append(signal_synchrony(spikes, settle=args.settle).clamp_min(0).cpu())
            if core.last_component_spikes is not None:
                # synchrony per component, combined after, which is what the
                # phase readout does and what the component collapse was losing
                per = torch.stack([
                    signal_synchrony(core.last_component_spikes[:, d],
                                     settle=args.settle).clamp_min(0)
                    for d in range(core.last_component_spikes.size(1))
                ])
                spike_parts.append(per.prod(dim=0).cpu())
            trains.append(spikes[:, :, args.settle:].cpu())
    plv, spike_sync, trains = torch.cat(plv), torch.cat(spike_sync), torch.cat(trains)
    plv_parts = torch.cat(plv_parts)
    spike_parts = torch.cat(spike_parts) if spike_parts else None

    def truth(i):
        l = labels[i]
        if l.max() < 1:
            return None
        g = l.clone(); g[g < 0] = g.max() + 1
        return torch.unique(g, return_inverse=True)[1]

    def report(name, pred_of):
        aris, fg_aris, ious = [], [], []
        for i in range(len(names)):
            g = truth(i)
            if g is None:
                continue
            p = pred_of(i)
            aris.append(adjusted_rand_index(p, g))
            ious.extend(foreground_iou(p, g))
            # Object-discovery papers report ARI over foreground only. Background
            # is 92% of this grid, so scoring it as one more cluster rewards
            # separating figure from ground and says little about whether one
            # object was told apart from another. Reported alongside, because the
            # repository's own history is in the all-patch number.
            fg = labels[i] >= 0
            if fg.sum() > 1 and labels[i][fg].unique().numel() > 1:
                fg_aris.append(adjusted_rand_index(p[fg], labels[i][fg]))
        print("%-40s %-9.4f %-9.4f %.4f"
              % (name, sum(aris) / len(aris),
                 sum(fg_aris) / max(len(fg_aris), 1), sum(ious) / len(ious)))

    counts = [int(truth(i).max()) + 1 for i in range(len(names)) if truth(i) is not None]
    recovered = sum(int(labels[i].max()) + 1 for i in range(len(names)) if labels[i].max() >= 0)
    if objects_by_name is None:
        print("dataset segmentation: %.2f objects and %.1f%% foreground per image"
              % (recovered / len(names), 100 * float((labels >= 0).float().mean())))
    else:
        declared = sum(len(objects_by_name[nm]) for nm in names)
        print("colour-derived masks recover %d of %d scene objects (%.0f%%); anything "
              "missing is scored as an error" % (recovered, declared, 100 * recovered / declared))
    print("checkpoint %s  (graph_top_k=%d)" % (Path(args.checkpoint), args.graph_top_k))
    print("oracle cluster count: mean %.1f, range %d-%d\n"
          % (sum(counts) / len(counts), min(counts), max(counts)))
    print("%-40s %-9s %-9s %s" % ("READOUT", "ARI", "FG-ARI", "fgIoU"))
    print("-" * 70)
    report("phases -> PLV -> spectral, oracle k",
           lambda i: spectral_cluster(plv[i], int(truth(i).max()) + 1))
    report("phases -> per-component PLV -> spectral, oracle k",
           lambda i: spectral_cluster(plv_parts[i], int(truth(i).max()) + 1))
    report("spikes -> correlation -> spectral, oracle k",
           lambda i: spectral_cluster(spike_sync[i], int(truth(i).max()) + 1))
    for k in args.fixed_k:
        report("phases -> spectral, k=%d" % k, lambda i, k=k: spectral_cluster(plv[i], k))
    for k in args.fixed_k:
        report("phases -> per-component PLV, k=%d" % k,
               lambda i, k=k: spectral_cluster(plv_parts[i], k))
    for k in args.fixed_k:
        report("spikes -> k-means on trains, k=%d" % k, lambda i, k=k: kmeans(trains[i], k))
    if spike_parts is not None:
        report("spikes -> per-component synchrony, oracle k",
               lambda i: spectral_cluster(spike_parts[i], int(truth(i).max()) + 1))
        for k in args.fixed_k:
            report("spikes -> per-component synchrony, k=%d" % k,
                   lambda i, k=k: spectral_cluster(spike_parts[i], k))
    # The control the README insists on: cluster the features the oscillators are
    # driven by, with no dynamics at all. Run in this program rather than a
    # separate script, because a standalone version of this comparison produced
    # numbers that did not reproduce here.
    report("CONTROL: gamma features, clustered directly",
           lambda i: kmeans(gamma[i].t().cpu(), int(truth(i).max()) + 1))
    report("CONTROL: chance", lambda i: torch.randint(0, int(truth(i).max()) + 1, (labels.size(1),)))

if __name__ == "__main__":
    main()
