import json
import unittest
from pathlib import Path

from evaluation.score import score

KEY = json.loads((Path(__file__).resolve().parents[1] / "evaluation" / "devops_tests_answer_key.json")
                 .read_text(encoding="utf-8"))


def _v(rule_id, category, file, line):
    return {"rule_id": rule_id, "category": category, "severity": "high", "file": file, "line": line,
            "description": "d", "evidence": "e", "recommendation": "r"}


class ScoreTests(unittest.TestCase):
    def test_precision_recall_and_duplicates(self):
        report = {"violations": [
            _v("SEC-001", "security", "app/config.py", 8),       # E01
            _v("SEC-001", "security", "app/config.py", 8),       # duplicate -> earns nothing
            _v("CI-002", "ci", ".github/workflows/ci.yml", 5),   # E11
            _v("HYG-001", "repository_hygiene", None, None),     # false positive (README exists)
        ]}
        r = score(report, KEY)
        self.assertEqual(len(r["true_positives"]), 2)
        self.assertEqual(len(r["false_positives"]), 2)
        self.assertAlmostEqual(r["precision"], 0.5)
        self.assertAlmostEqual(r["recall"], 2 / len(KEY["expected"]))

    def test_line_tolerance(self):
        far = {"violations": [_v("CFG-001", "configuration", "app/config.py", 40)]}
        self.assertEqual(score(far, KEY)["true_positives"], [])


if __name__ == "__main__":
    unittest.main()
