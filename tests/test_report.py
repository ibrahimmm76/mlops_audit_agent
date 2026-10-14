import json
import tempfile
import unittest
from pathlib import Path

from audit_agent.report import (
    FindingError,
    FindingStore,
    build_finding,
    build_report,
    validate_report,
    write_report,
)
from tests.helpers import FAKE_AWS_KEY

VALID = {
    "rule_id": "SEC-001", "category": "security", "severity": "critical", "file": "config.py", "line": 17,
    "description": "Hardcoded API credential", "evidence": "Credential assigned directly in source code",
    "recommendation": "Use a secrets manager or environment variable",
}


class FindingValidationTests(unittest.TestCase):
    def test_valid_finding(self):
        f = build_finding(VALID, file_exists=lambda p: True, line_count=lambda p: 50)
        self.assertEqual(f.to_dict(), VALID)

    def test_null_file_and_line(self):
        f = build_finding({**VALID, "file": "", "line": 0})
        self.assertIsNone(f.file)
        self.assertIsNone(f.line)
        f = build_finding({**VALID, "file": None, "line": None})
        self.assertIsNone(f.file)

    def test_category_aliases(self):
        f = build_finding({**VALID, "category": "GitHub Actions"})
        self.assertEqual(f.category, "ci")

    def test_rejections(self):
        bad_cases = [
            {**VALID, "severity": "urgent"},
            {**VALID, "rule_id": "security-1"},
            {**VALID, "category": "ml"},
            {**VALID, "file": None, "line": 3},
            {**VALID, "file": "../x.py"},
            {**VALID, "line": -1},
            {**VALID, "description": ""},
        ]
        for case in bad_cases:
            with self.assertRaises(FindingError, msg=case):
                build_finding(case)

    def test_file_must_exist_and_line_in_range(self):
        with self.assertRaises(FindingError):
            build_finding(VALID, file_exists=lambda p: False)
        with self.assertRaises(FindingError):
            build_finding(VALID, file_exists=lambda p: True, line_count=lambda p: 5)

    def test_leading_dot_paths_are_preserved(self):
        f = build_finding({**VALID, "file": "./.env", "line": 1})
        self.assertEqual(f.file, ".env")

    def test_secret_values_never_in_output(self):
        f = build_finding({**VALID, "evidence": f"API_KEY = {FAKE_AWS_KEY} in config"},
                          known_secret_values=frozenset({FAKE_AWS_KEY}))
        self.assertNotIn(FAKE_AWS_KEY, json.dumps(f.to_dict()))

    def test_deduplication(self):
        store = FindingStore()
        store.add(build_finding(VALID))
        with self.assertRaises(FindingError):
            store.add(build_finding(VALID))
        store.add(build_finding({**VALID, "line": 20, "description": "Another credential"}))
        self.assertEqual(len(store), 2)
        store.withdraw("F1")
        self.assertEqual(len(store), 1)
        with self.assertRaises(FindingError):
            store.withdraw("F99")


class ReportTests(unittest.TestCase):
    def test_report_contract_and_ordering(self):
        low = build_finding({**VALID, "severity": "low", "rule_id": "HYG-003", "category": "repository_hygiene"})
        crit = build_finding(VALID)
        report = build_report("https://github.com/example/demo-project", [low, crit])
        self.assertEqual(validate_report(report), [])
        self.assertEqual([v["severity"] for v in report["violations"]], ["critical", "low"])
        self.assertEqual(set(report), {"repository", "violations"})

    def test_empty_report_is_valid(self):
        report = build_report("https://github.com/a/b", [])
        self.assertEqual(report["violations"], [])
        self.assertEqual(validate_report(report), [])

    def test_validate_report_detects_problems(self):
        self.assertTrue(validate_report({"repository": "x", "violations": [{"severity": "bad"}]}))
        self.assertTrue(validate_report([]))

    def test_write_report_atomic_valid_json(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "sub" / "report.json"
            write_report(out, build_report("https://github.com/a/b", [build_finding(VALID)]))
            data = json.loads(out.read_text())
            self.assertEqual(data["violations"][0]["rule_id"], "SEC-001")
            with self.assertRaises(ValueError):
                write_report(out, {"violations": "nope"})


if __name__ == "__main__":
    unittest.main()
