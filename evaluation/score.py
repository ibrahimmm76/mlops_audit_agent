#!/usr/bin/env python3
"""Score an agent report against an answer key (precision / recall).

    python evaluation/score.py sample_report.json evaluation/devops_tests_answer_key.json

A reported violation matches an expected defect when:
  * the file matches (expected file may be a list of alternatives; null = repository-level), and
  * its rule_id is one the key accepts (a custom id ending in 090-099 may match by category), and
  * when both sides give a line, they are within +/-3 lines.
Each expected defect can be matched by at most one reported violation (duplicates earn nothing),
mirroring the assignment's grading. Unmatched reports count as false positives - review them by
hand: some may be genuine issues the key does not list.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LINE_TOLERANCE = 3


def _files(expected: dict) -> list[str | None]:
    f = expected.get("file")
    return f if isinstance(f, list) else [f]


def matches(report_item: dict, expected: dict) -> bool:
    if report_item.get("file") not in _files(expected):
        return False
    rule_id = str(report_item.get("rule_id", ""))
    custom = rule_id[-3:].startswith("09")
    if rule_id not in expected["rule_ids"] and not (custom and report_item.get("category") in expected["categories"]):
        return False
    rl, el = report_item.get("line"), expected.get("line")
    if rl is not None and el is not None and abs(int(rl) - int(el)) > LINE_TOLERANCE:
        return False
    return True


def score(report: dict, key: dict) -> dict:
    remaining = list(key["expected"])
    tp, fp = [], []
    for item in report.get("violations", []):
        candidates = [e for e in remaining if matches(item, e)]
        # prefer the expected defect whose line is closest to the reported one
        hit = min(candidates, key=lambda e: abs((item.get("line") or 0) - (e.get("line") or 0)), default=None)
        if hit:
            remaining.remove(hit)
            tp.append((item, hit))
        else:
            fp.append(item)
    n_rep = len(report.get("violations", []))
    n_exp = len(key["expected"])
    return {
        "true_positives": tp,
        "false_positives": fp,
        "missed": remaining,
        "precision": len(tp) / n_rep if n_rep else 1.0,
        "recall": len(tp) / n_exp if n_exp else 1.0,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    report = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    key = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
    r = score(report, key)
    print(f"Reported: {len(report.get('violations', []))}   Expected: {len(key['expected'])}")
    print(f"Precision: {r['precision']:.2f}   Recall: {r['recall']:.2f}\n")
    print("Matched:")
    for item, exp in r["true_positives"]:
        print(f"  [ok] {exp['id']} {exp['summary']}  <-  {item['rule_id']} {item['file']}:{item['line']}")
    print("\nUnmatched reports (false positives, or real issues not in the key):")
    for item in r["false_positives"] or []:
        print(f"  [fp] {item['rule_id']} {item['file']}:{item['line']} - {item['description']}")
    print("\nMissed expected defects:")
    for exp in r["missed"]:
        print(f"  [--] {exp['id']} {exp['summary']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
