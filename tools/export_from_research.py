"""One-time migration, run in the original research environment; not a runtime dependency."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    root = args.legacy_root.resolve()
    sys.path[:0] = [str(root), str(root / 'snn_kuramoto_bidirectional')]
    import numpy as np
    import torch
    from collaborative_test.SW_0249_four_model_35000 import factory
    from collaborative_test.SW_0094_aligned_joint_pilot.run import hparams
    from collaborative_test.SW_0154_native64_resolution_compare.native64_model import configure_grid_hparams
    torch.set_num_threads(1)
    p = factory.prepare('240', args.seed)
    found = p['found']
    folder = factory.archive('240', args.seed)
    report = json.loads((folder / 'training_report.json').read_bytes())
    factory.load_weights(found, folder / 'encoder_work/boundary.ptx', report['checkpoint_sha256'])
    assert factory.model_sha(found) == report['final_model_sha256']
    hp = vars(configure_grid_hparams(hparams('raw'), grid_size=64, graph_top_k=512)).copy()
    assert hp['sc'] is None
    # State dictionaries do not contain runtime flags, geometry or method overrides.
    def simple(v):
        return v is None or type(v) in (bool, int, float, str) or (
            isinstance(v, (list, tuple)) and all(simple(x) for x in v))
    attrs = {name: {k: v for k, v in vars(module).items()
                    if not k.startswith('_') and k not in ('training', 'device') and simple(v)}
             for name, module in found.wrapped.named_modules()}
    config = dict(format_version=1, model='SW0240', endpoint='SW0249_matched35000_terminal',
                  seed=args.seed, grid_size=64, total_steps=1024, settle_steps=512,
                  integration_arm=found.wrapped.arm, hparams=hp, runtime_attributes=attrs,
                  geodesic=dict(effective_contrast=8.0, squared_dissimilarity=True, row_chunk=1),
                  model_sha256=report['final_model_sha256'],
                  source_checkpoint_sha256=report['checkpoint_sha256'],
                  feature_clip=float(found.feature_clip), mask_supervised=True)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f'sw240_seed{args.seed}.pt'
    if path.exists():
        raise FileExistsError(path)
    def cpu_state(module):
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    bundle = dict(config=config, wrapped_state=cpu_state(found.wrapped),
                  encoder_state=cpu_state(found.encoder),
                  feature_mean=torch.as_tensor(found.feature_mean).detach().cpu().clone(),
                  feature_std=torch.as_tensor(found.feature_std).detach().cpu().clone(),
                  teacher_encoder_state=cpu_state(p['teacher'].encoder),
                  teacher_projection_state=cpu_state(p['teacher'].wrapped.core.gamma_channel_proj),
                  teacher_model_sha256=factory.model_sha(p['teacher']))
    # The teacher and student differ only in encoder and sensory projection.
    for key, value in p['teacher'].wrapped.state_dict().items():
        if key != 'core.gamma_channel_proj.weight':
            assert torch.equal(value, found.wrapped.state_dict()[key]), key
    temp = path.with_suffix('.tmp')
    torch.save(bundle, temp)
    temp.replace(path)
    config_path = path.with_suffix('.json')
    config_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(dict(status='exported', file=str(path), bytes=path.stat().st_size,
                          sha256=sha, config=config)))


if __name__ == '__main__':
    main()
