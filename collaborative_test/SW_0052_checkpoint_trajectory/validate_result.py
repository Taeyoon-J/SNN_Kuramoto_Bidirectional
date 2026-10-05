import json
import math
import sys


def validate(path, checkpoint, count, steps, settle):
    report = json.load(open(path))
    assert report["checkpoint"] == checkpoint
    assert report["ids"] == [1320, 1320 + count - 1]
    assert report["images"] == count
    assert report["ground_truth_used_for_prediction"] is False
    assert report["inference"]["steps"] == steps
    assert report["inference"]["settle"] == settle
    rows = report["sweep"]
    assert len(rows) == 5
    assert [row["synchrony_threshold"] for row in rows] == [0.05, 0.1, 0.2, 0.35, 0.5]
    for row in rows:
        metrics = row["scored_targets"]["our_hdf5"]["metrics"]
        assert all(math.isfinite(float(metrics[key])) for key in
                   ("fg_ari", "foreground_iou", "matched_object_iou"))


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
