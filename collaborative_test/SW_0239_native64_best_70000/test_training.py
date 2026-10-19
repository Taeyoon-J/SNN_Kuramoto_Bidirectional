"""Exercise source selection, complete TRAIN order, and real Adam checkpoint resume."""
import copy
import json
from types import SimpleNamespace
import h5py
import numpy as np
import torch
from collaborative_test.SW_0239_native64_best_70000 import train, data


def test_full_history_order_rejects_duplicate_or_changed_middle_batch(monkeypatch):
    monkeypatch.setattr(data, 'STEPS_PER_EPOCH', 2)
    monkeypatch.setattr(data, 'BATCH', 2)
    monkeypatch.setattr(data, 'epoch_ids', lambda seed, epoch: np.arange(4))
    rows = [dict(update=i+1, TRAIN_ids=[(i % 2)*2, (i % 2)*2+1], loss=1., frequency_denoising=.5, gamma_anchor=.5, gradient_norm=1.) for i in range(4)]
    train.check_history(rows, 0)
    rows[1]['TRAIN_ids'] = [0, 1]
    try:
        train.check_history(rows, 0)
    except ValueError:
        pass
    else:
        raise AssertionError('repeated batch accepted')


def test_best_source_requires_audited_full320_and_freezes_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'ARCHIVE', tmp_path / 'child')
    monkeypatch.setattr(train, 'fingerprint', lambda: {'fixed': True})
    modules = {}
    for number, fg in ((227, .85), (237, .88)):
        folder = tmp_path / str(number); folder.mkdir()
        (folder / 'completion_audit.py').write_bytes(b'fixed auditor')
        module = SimpleNamespace(ARCHIVE=folder, HERE=folder, __name__=str(number), fingerprint=lambda: {'parent': True})
        score = dict(fingerprint={'parent': True}, image_count=320, seed_scope=[0,1,2],
            scores=dict(joint_projection=dict(fg_ari=fg, foreground_iou=.6, matched_object_iou=.7),
                        slot=dict(fg_ari=.78, foreground_iou=.2, matched_object_iou=.2)))
        train.write_json(folder / 'score_v1/summary.json', score)
        raw = (folder / 'score_v1/summary.json').read_bytes()
        audit = dict(score_sha256=train.sha(raw), diagnostic_source_sha256=train.sha(b'fixed auditor'),
            raw_masks_independently_rescored=2880, independent_metrics_means_and_bootstrap_maximum_abs_error=0.)
        train.write_json(folder / 'completion_audit_v1/summary.json', audit)
        modules[number] = module
    monkeypatch.setattr(train.importlib, 'import_module', lambda name: modules[237 if '0237' in name or name == '237' else 227])
    chosen = train.select_source()
    assert chosen['selected']['experiment'] == 237
    assert train.select_source() == chosen
    # An evidence change after registration must fail, never silently reselect.
    (modules[237].ARCHIVE / 'score_v1/summary.json').write_bytes(b'changed')
    try:
        train.selection()
    except ValueError:
        pass
    else:
        raise AssertionError('changed selected evidence accepted')


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_encoder = torch.nn.Linear(3, 8)
        self.context = torch.nn.Linear(8, 8)
    def forward(self, x):
        return self.context(self.base_encoder(x.float())).unsqueeze(-1)


def fake_found():
    torch.manual_seed(239)
    core = torch.nn.Module(); core.gamma_channel_proj = torch.nn.Linear(8, 4, bias=False)
    wrapped = torch.nn.Module(); wrapped.core = core
    encoder = Encoder()
    return SimpleNamespace(wrapped=wrapped, encoder=encoder, encode=encoder,
        feature_mean=np.zeros(1), feature_std=np.ones(1), feature_clip=1.)


def test_real_Adam_checkpoint_resume_matches_uninterrupted_updates(tmp_path, monkeypatch):
    dataset = tmp_path / 'train.h5'
    with h5py.File(dataset, 'w') as stream:
        stream['image'] = np.arange(12, dtype=np.float32).reshape(4, 3) / 10
    stat = dataset.stat()
    monkeypatch.setattr(data, 'BATCH', 2); monkeypatch.setattr(data, 'EPOCHS', 2)
    monkeypatch.setattr(data, 'TRAIN_COUNT', 4); monkeypatch.setattr(data, 'STEPS_PER_EPOCH', 2)
    monkeypatch.setattr(data, 'TOTAL_STEPS', 4)
    monkeypatch.setattr(data, 'epoch_ids', lambda seed, epoch: np.arange(4))
    def read(stream, ids):
        return np.asarray(stream['image'][ids]), np.zeros((len(ids), 1, 1), dtype=np.int64)
    monkeypatch.setattr(data, 'read_batch', read)
    monkeypatch.setattr(train.helpers, 'effective_frequency', lambda core, gamma: core.gamma_channel_proj(gamma.transpose(1, 2)))
    model_sha = train.source.source.full.screen.native._state_sha
    def prepared():
        found = fake_found(); teacher = fake_found()
        teacher.encoder.requires_grad_(False); teacher.wrapped.requires_grad_(False)
        return dict(found=found, teacher=teacher, optimizer=train.optimizer_for(found), seed=0,
            binding=dict(dataset=dict(path=str(dataset), size=stat.st_size, mtime_ns=stat.st_mtime_ns),
                frozen_nonprojection_core_sha256=train.projection.frozen_core_sha(found),
                input_statistics_sha256=train.helpers.statistics_sha(found),
                teacher_model_sha256=model_sha(teacher.wrapped, teacher.encoder)))
    runtime = train.checkpoint.runtime_signature('cpu')
    full = prepared(); root = tmp_path / 'a'
    monkeypatch.setattr(train, 'ARCHIVE', root)
    train.write_json(train.archive_for(0) / 'admission.json', dict(binding=full['binding']))
    expected = train.fit(full, 'cpu', runtime)
    interrupted = prepared(); root = tmp_path / 'b'
    monkeypatch.setattr(train, 'ARCHIVE', root)
    train.write_json(train.archive_for(0) / 'admission.json', dict(binding=interrupted['binding']))
    original = train.teacher_targets; calls = []
    def stop_on_third(*args):
        calls.append(1)
        if len(calls) == 3:
            raise InterruptedError('simulated interruption after committed epoch1')
        return original(*args)
    monkeypatch.setattr(train, 'teacher_targets', stop_on_third)
    try:
        train.fit(interrupted, 'cpu', runtime)
    except InterruptedError:
        pass
    else:
        raise AssertionError('fixture did not interrupt')
    assert train.checkpoint.inspect_committed_metadata(train.archive_for(0) / 'encoder_work')['completed_updates'] == 2
    monkeypatch.setattr(train, 'teacher_targets', original)
    resumed = prepared(); actual = train.fit(resumed, 'cpu', runtime)
    assert actual['actual_Adam_steps'] == [4]
    for key in ('final_model_sha256', 'final_encoder_sha256', 'history_sha256'):
        assert actual[key] == expected[key]
    assert all(torch.equal(a, b) for a, b in zip(full['found'].encoder.parameters(), resumed['found'].encoder.parameters()))
