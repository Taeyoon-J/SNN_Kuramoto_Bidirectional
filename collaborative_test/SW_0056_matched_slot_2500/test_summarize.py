import unittest

from summarize import summarize_reports


def report(seed, ari=None):
    return {
        "ground_truth_used_for_prediction": False,
        "scores": {"mean": {"fg_ari": seed / 10 if ari is None else ari,
                            "foreground_iou": .3, "matched_object_iou": .4}},
        "protocol": {
            "seed": seed, "image_ids": [1320, 1639], "count": 320, "inference_seed": 0,
            "resolution": [128, 128], "num_slots": 11, "iterations": 3,
            "checkpoint_dir": f"/models/seed{seed}/checkpoint",
            "checkpoint_prefix": f"/models/seed{seed}/checkpoint/ckpt-1563",
            "checkpoint_sha256": str(seed) * 64, "model_sha256": "a" * 64,
            "training_protocol_path": f"/models/seed{seed}/training_protocol.json",
            "training_protocol": {"seed": seed, "validation_ids_inclusive": [1320, 1639],
                                  "optimizer_updates": 1563, "effective_image_exposures": 25000},
        },
    }


class SummaryTests(unittest.TestCase):
    def test_three_seed_mean_and_std(self):
        result = summarize_reports({seed: report(seed) for seed in (0, 1, 2)})
        self.assertAlmostEqual(result["metrics"]["fg_ari"]["mean"], .1)
        self.assertEqual(result["seeds"], [0, 1, 2])

    def test_missing_seed_mismatch_or_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            summarize_reports({0: report(0), 1: report(1)})
        bad = {seed: report(seed) for seed in (0, 1, 2)}
        bad[2]["protocol"]["iterations"] = 4
        with self.assertRaises(ValueError):
            summarize_reports(bad)
        bad = {seed: report(seed) for seed in (0, 1, 2)}
        bad[0]["scores"]["mean"]["fg_ari"] = float("nan")
        with self.assertRaises(ValueError):
            summarize_reports(bad)

    def test_cpu_gpu_execution_provenance_does_not_split_model_protocol_family(self):
        reports = {seed: report(seed) for seed in (0, 1, 2)}
        reports[0]["protocol"]["training_protocol"].update(
            trainer_sha256="cpu", execution_backend="cpu_forced")
        for seed in (1, 2):
            reports[seed]["protocol"]["training_protocol"].update(
                trainer_sha256="gpu", execution_backend="gpu_opt_in")
        result = summarize_reports(reports)
        self.assertEqual(result["seeds"], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
