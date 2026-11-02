"""Native64 core + transferred SW0260 context readout; no output upsampling."""
import importlib.util
import types
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


class Native64Foundation:
    """Expose the same foundation interface without native32 shape assertions."""
    def __init__(self, parent, wrapped, patcher):
        self.seed = parent.seed
        self.wrapped, self.patcher = wrapped, patcher
        self.encoder = parent.encoder
        self.feature_mean, self.feature_std = parent.feature_mean, parent.feature_std
        self.feature_clip = parent.feature_clip
        self.provenance = dict(parent.provenance, runtime_core_grid=64,
                               native64_training_updates=0)

    def encode(self, rgb):
        if rgb.dtype != torch.uint8 or rgb.ndim != 4 or rgb.shape[1:] != (128,128,3):
            raise ValueError('uint8 RGB[B,128,128,3] required')
        device = next(self.encoder.parameters()).device
        maps = self.encoder(rgb.to(device).permute(0,3,1,2).float()/255.)
        return self.patcher(((maps-self.feature_mean)/self.feature_std).clamp(
            -self.feature_clip, self.feature_clip))

    def rollout(self, gamma, *, total_steps=1024, live_tail_steps=0):
        from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
        return late_rollout(self.wrapped, gamma, total_steps=total_steps,
                            live_tail_steps=live_tail_steps)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def load(checkpoint, device='cpu', *, trainable=False):
    context = module('native64_parent_portable', HERE.parent/'readout64_optimization/portable.py')
    found, head, record = context.load(checkpoint, device)
    parent = HERE.parent/'unsupervised_readout_benchmark/checkpoints'/f"sw0259_seed{record['seed']}.pt"
    blob = torch.load(parent, map_location='cpu', weights_only=True)
    implementation = module('native64_implementation', HERE/'model.py')
    wrapped = implementation.build_wrapped(blob, device)
    # Release the 1024-node core; the input encoder and its statistics are reused.
    from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
    found = Native64Foundation(found, wrapped, FeaturePatchGammaInitializer(grid_size=64).to(device))
    head.routing = types.MethodType(implementation.native_routing, head)
    wrapped.requires_grad_(trainable).train(trainable)
    head.requires_grad_(trainable).train(trainable)
    return found, head, dict(record, core_grid=64, nodes=4096, graph_top_k=512,
                            native64_training_updates=0, status='transferred_initialization')


def encode(found, rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (128, 128, 3):
        raise ValueError('exact uint8 RGB128x128 required')
    device = next(found.encoder.parameters()).device
    image = torch.from_numpy(rgb[None].copy()).to(device)
    gamma = found.encode(image)
    if gamma.shape != (1, 8, 4096):
        raise AssertionError('native64 gamma required')
    return gamma


@torch.no_grad()
def predict(found, head, rgb):
    from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
    from collaborative_test.SW_0162_connected_signed_partition import preflight as source
    from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
    gamma = encode(found, rgb)
    trace, a = source.capture_source_coupling(found.wrapped.core.kuramoto,
        lambda: late_rollout(found.wrapped, gamma, total_steps=1024, live_tail_steps=0))
    q_tensor = spike_synchrony_affinity(trace['spikes'], trace['component_spikes'],
                                      settle=512, affinity_mode='spike')
    if q_tensor.shape != (1,4096,4096) or not torch.isfinite(q_tensor).all():
        raise ValueError('finite native64 spike affinity required')
    q = q_tensor[0].cpu().numpy().astype(np.float64)
    gain = float(found.wrapped.core.graph_generator.log_coupling_gain.exp().cpu())
    solver = module('native64_partition', HERE/'partition.py')
    regions = solver.foreground_induced_merge(q, a[0].cpu().numpy().astype(np.float64),
        gain, np.ones((64, 64), dtype=np.int64), grid_shape=(64, 64))['labels']
    vendor = HERE.parent/'vendor'
    activity = module('native64_activity', vendor/'readout_benchmark/head.py')
    candidates = module('native64_candidates', vendor/'neutral_readout/head.py')
    features = activity.activity_features(trace)[0]
    masks, prior = candidates.candidates(features, regions)
    return head.predict(dict(features=features, masks=masks, prior=prior,
                             gamma=gamma[0].T)).cpu().numpy()


if __name__ == '__main__':
    import argparse
    import json
    from PIL import Image
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    torch.set_num_threads(1)
    found, head, record = load(args.checkpoint, args.device)
    labels = predict(found, head, np.asarray(Image.open(args.image).convert('RGB')))
    np.save(args.output, labels)
    print(json.dumps(dict(output=args.output, shape=list(labels.shape), provenance=record)))
