import math

import pytest

from compare import compare


def summaries(model_values=(.8, .5, .6), slot_values=(.7, .4, .5)):
    model = {
        "experiment": "SW0072 frozen trained graph full mean",
        "means": dict(zip(("fg_ari", "foreground_iou", "matched_object_iou"), model_values)),
    }
    slot = {
        "experiment": "SW0056 matched-data Slot Attention three-seed summary",
        "image_ids_inclusive": [1320, 1639], "count": 320, "seeds": [0, 1, 2],
        "ground_truth_used_for_prediction": False,
        "metrics": {key: {"mean": value} for key, value in zip(
            ("fg_ari", "foreground_iou", "matched_object_iou"), slot_values)},
    }
    return model, slot


def test_all_three_strict_improvement_required():
    model, slot = summaries()
    result = compare(model, slot)
    assert result["goal_achieved"] is True
    assert all(row["strictly_exceeds_slot"] for row in result["metrics"].values())

    model, slot = summaries(model_values=(.7, .5, .6))
    result = compare(model, slot)
    assert result["goal_achieved"] is False
    assert result["metrics"]["fg_ari"]["strictly_exceeds_slot"] is False


def test_contract_and_finite_values_are_enforced():
    model, slot = summaries()
    slot["image_ids_inclusive"] = [1000, 1319]
    with pytest.raises(ValueError, match="split"):
        compare(model, slot)
    model, slot = summaries(model_values=(math.nan, .5, .6))
    with pytest.raises(ValueError, match="non-finite"):
        compare(model, slot)
