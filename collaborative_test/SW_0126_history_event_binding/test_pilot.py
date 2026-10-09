"""Focused CPU contracts for the conditional paired SW0126 pilot."""
import numpy as np
import torch

from collaborative_test.SW_0126_history_event_binding import pilot_evaluate, pilot_queue, pilot_run


def test_slot_readout_largest_background_and_singleton_omission():
    labels = torch.zeros((1, 11, 256))
    labels[:, 0, :130] = 1
    labels[:, 3, 130:255] = 2
    labels[:, 7, 255] = 3
    result = pilot_evaluate.slot_labels_from_assignments(labels)
    assert result.shape == (1, 16, 16)
    assert torch.all(result.reshape(-1)[:130] == 0)  # largest slot is background
    assert torch.all(result.reshape(-1)[130:255] == 4)
    assert result.reshape(-1)[255] == 0  # singleton is omitted


def test_equal_largest_slot_tie_uses_lowest_id_as_background():
    labels = torch.zeros((1, 11, 256))
    labels[:, 1, :128] = 1
    labels[:, 2, 128:] = 1
    result = pilot_evaluate.slot_labels_from_assignments(labels)
    assert torch.all(result.reshape(-1)[:128] == 0)
    assert torch.all(result.reshape(-1)[128:] == 3)


def test_paired_bootstrap_is_fixed_paired_seeded_and_rejects_bad_contract():
    candidate = np.full(320, 0.7)
    reference = np.full(320, 0.2)
    first = pilot_evaluate.paired_bootstrap(candidate, reference)
    second = pilot_evaluate.paired_bootstrap(candidate, reference)
    assert first == second
    assert first["ci95"][0] > 0 and abs(first["mean_delta"] - 0.5) < 1e-12
    try:
        pilot_evaluate.paired_bootstrap(candidate[:319], reference[:319])
    except ValueError:
        pass
    else:
        raise AssertionError("paired bootstrap accepted an incomplete evaluation set")


def test_separate_core_and_head_adam_update_changes_both_groups():
    core = torch.nn.Parameter(torch.tensor([0.25, -0.5]))
    head = torch.nn.Parameter(torch.tensor([0.75]))
    core_opt = torch.optim.Adam([core], lr=pilot_run.CORE_LR)
    head_opt = torch.optim.Adam([head], lr=pilot_run.HEAD_LR)
    before_core, before_head = core.detach().clone(), head.detach().clone()
    loss = ((core.sum() * head[0]) - 1.0) ** 2
    summary = pilot_run._apply_two_optimizer_step(loss, [core], [head], core_opt, head_opt)
    assert summary["core"]["finite"] and summary["head"]["finite"]
    assert summary["core"]["norm"] > 0 and summary["head"]["norm"] > 0
    assert not torch.equal(core, before_core)
    assert not torch.equal(head, before_head)


def test_native_theta_layout_matches_custom_full_history_exactly():
    custom = torch.arange(2 * 5 * 4 * 13, dtype=torch.float32).reshape(2, 5, 4, 13)
    native = custom.permute(0, 3, 1, 2).contiguous()  # production [B,T,N,D]
    assert pilot_run.theta_trace_exact(custom, native)
    native[0, 0, 0, 0] += 1
    assert not pilot_run.theta_trace_exact(custom, native)


def test_history_event_binder_receives_emitted_spike_not_binary_event():
    event = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    gate = torch.tensor([[[[0.25, 0.75], [0.50, 0.40]]]])
    emitted = gate * event
    history_input = pilot_run.binder_trace_for_arm(emitted, gate, "history_event")
    gate_input = pilot_run.binder_trace_for_arm(emitted, gate, "gate_only")
    assert torch.equal(history_input, emitted)
    assert torch.equal(gate_input, gate)
    assert not torch.equal(history_input, event)


def test_pair_queue_orders_all_screen_proofs_before_both_trains_and_eval():
    tasks = pilot_queue.task_plan()
    preflights = [t for t in tasks if t["stage"] == "preflight"]
    trains = [t for t in tasks if t["stage"] == "train"]
    evaluation = [t for t in tasks if t["stage"] == "evaluate"]
    assert len(preflights) == 2 and len(trains) == 2 and len(evaluation) == 1
    assert all(len(t["depends_on"]) == 3 for t in preflights)
    assert {t["arm"] for t in preflights} == set(pilot_run.ARM_NAMES)
    assert {t["arm"] for t in trains} == set(pilot_run.ARM_NAMES)
    assert len(evaluation[0]["depends_on"]) == 2
    assert pilot_queue.gpu_is_exclusive(512, [])
    assert not pilot_queue.gpu_is_exclusive(513, [])
    assert not pilot_queue.gpu_is_exclusive(10, [123])


def test_metric_contract_rejects_nan_and_wrong_count():
    score = {metric: {"mean": 0.5, "valid_count": 320,
                      "per_image": [0.5] * 320} for metric in pilot_evaluate.METRICS}
    assert all(len(v) == 320 for v in pilot_evaluate._finite_metric_rows(score).values())
    score["fg_ari"]["per_image"][7] = float("nan")
    try:
        pilot_evaluate._finite_metric_rows(score)
    except ValueError:
        pass
    else:
        raise AssertionError("metric gate accepted a nonfinite per-image score")


def test_pilot_gate_requires_both_paired_anchors_and_registered_slot_margin():
    def score(fg, fiou, miou):
        return {metric: {"mean": value, "valid_count": 320,
                         "per_image": [value] * 320}
                for metric, value in (("fg_ari", fg), ("foreground_iou", fiou),
                                      ("matched_object_iou", miou))}
    result = pilot_evaluate._promotion(
        score(0.8, 0.30, 0.31), score(0.7, 0.2, 0.2),
        {"fg_ari": np.full(320, 0.6), "foreground_iou": np.full(320, 0.2),
         "matched_object_iou": np.full(320, 0.2)},
        pilot_evaluate.SLOT_IOU_REFERENCE)
    assert result["gates"]["pilot_pass"] is True
    blocked = pilot_evaluate._promotion(
        score(0.8, 0.25, 0.31), score(0.7, 0.2, 0.2),
        {"fg_ari": np.full(320, 0.6), "foreground_iou": np.full(320, 0.2),
         "matched_object_iou": np.full(320, 0.2)},
        pilot_evaluate.SLOT_IOU_REFERENCE)
    assert blocked["gates"]["pilot_pass"] is False
    assert blocked["gates"]["foreground_iou_gt_slot_plus_005"] is False


def test_rgb_objective_is_exact_native_pixel_mse_without_extra_terms():
    prediction = torch.tensor([[[[0.0, 1.0]]]])
    target = torch.tensor([[[[1.0, 0.0]]]])
    loss = pilot_run.native_rgb_objective(prediction, target)
    assert loss.item() == 1.0
    assert pilot_run.CORE_LR == 3e-5 and pilot_run.HEAD_LR == 1e-3
    assert pilot_run.UPDATES == 256 and pilot_run.BATCH == 16
