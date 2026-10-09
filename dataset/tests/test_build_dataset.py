import csv
import json
import tempfile
import unittest
from pathlib import Path
from dataset.build_dataset import DEFAULT, action_for, bkt_step, build, split_for, validate_config

class DatasetTests(unittest.TestCase):
    def test_bkt_moves_consistent_evidence(self):
        p = DEFAULT["bkt"]
        self.assertGreater(bkt_step(.5, True, p), .5)
        self.assertLess(bkt_step(.5, False, p), .5)

    def test_policy_is_based_on_pre_state_and_prior_errors(self):
        self.assertEqual(action_for(.2, [], DEFAULT), "TEACH_PRIOR")
        self.assertEqual(action_for(.6, [], DEFAULT), "HINT")
        self.assertEqual(action_for(.9, [1, 1], DEFAULT), "ASK")
        self.assertEqual(action_for(.9, [0, 1], DEFAULT), "ANSWER")

    def test_student_split_is_stable(self):
        self.assertEqual(split_for("student-1", DEFAULT), split_for("student-1", DEFAULT))

    def test_build_preserves_observed_and_estimated_labels(self):
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td)/"raw.csv", Path(td)/"out.csv"
            src.write_text("user_id,problem_id,skill_id,correct,order_id\ns1,p1,k1,1,1\ns1,p2,k1,0,2\n", encoding="utf-8")
            meta = build(src, dst)
            with dst.open(encoding="utf-8", newline="") as f: rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2); self.assertEqual(rows[0]["correct"], "1")
            self.assertNotEqual(rows[0]["mastery_pre"], rows[0]["mastery_post"])
            self.assertEqual(meta["output_rows"], 2)

    def test_bad_splits_rejected(self):
        cfg = json.loads(json.dumps(DEFAULT)); cfg["split"]["test"] = .2
        with self.assertRaises(ValueError): validate_config(cfg)

if __name__ == "__main__": unittest.main()
