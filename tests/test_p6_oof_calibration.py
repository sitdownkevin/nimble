import json
import tempfile
import unittest
from pathlib import Path

from nimble.training.oof_calibration import (
    aggregate_oof_reports,
    apply_locked_temperatures,
    blend_centered_reports,
    summarize_rows,
)


class P6OOFCalibrationTests(unittest.TestCase):
    def test_aggregate_oof_and_locked_holdout_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for fold in range(3):
                rows = []
                for p in range(4):
                    for idx, (target, correct) in enumerate([(True, p < 3), (False, p < 3)]):
                        logit_t = 5.0 if (target == correct) else -5.0
                        rows.append({
                            "id": f"f{fold}_p{p}_{idx}",
                            "family": f"dom_{fold}",
                            "pair_id": f"f{fold}_p{p}",
                            "kind": "noul",
                            "target": target,
                            "correct": correct,
                            "logits": {"false": -logit_t, "true": logit_t},
                            "probabilities": {"false": 0.0067 if logit_t > 0 else 0.9933,
                                              "true": 0.9933 if logit_t > 0 else 0.0067},
                            "nll": 0.0067 if correct else 10.0,
                            "brier": 0.0001 if correct else 1.98,
                        })
                fp = Path(tmp) / f"fold{fold}.json"
                fp.write_text(json.dumps({"rows": rows}))
                paths[fold] = fp

            agg = aggregate_oof_reports(paths)
            self.assertEqual(agg["total_oof_rows"], 24)
            self.assertGreater(agg["locked_oof_temperatures"]["noul"], 2.0)
            self.assertLess(
                agg["oof_nested_calibrated_summary"]["all"]["nll"],
                agg["oof_uncalibrated_summary"]["all"]["nll"],
            )
            # Apply locked temperature to a holdout fold without refitting
            holdout = json.loads(paths[0].read_text())["rows"]
            cal = apply_locked_temperatures(holdout, agg["locked_oof_temperatures"])
            self.assertLess(summarize_rows(cal)["all"]["nll"], summarize_rows(holdout)["all"]["nll"])

    def test_blend_centered_reports(self):
        r1 = {"rows": [
            {"id": "1", "family": "d", "pair_id": "p", "kind": "noul", "target": True, "logits": {"false": 2.0, "true": 1.0}},
            {"id": "2", "family": "d", "pair_id": "p", "kind": "noul", "target": False, "logits": {"false": 3.0, "true": 0.0}},
        ]}
        r2 = {"rows": [
            {"id": "1", "family": "d", "pair_id": "p", "kind": "noul", "target": True, "logits": {"false": 0.0, "true": 4.0}},
            {"id": "2", "family": "d", "pair_id": "p", "kind": "noul", "target": False, "logits": {"false": 0.0, "true": 1.0}},
        ]}
        bl = blend_centered_reports([r1, r2])
        self.assertEqual(bl["summary"]["all"]["accuracy"], 1.0)
        self.assertEqual(bl["summary"]["pairs"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
