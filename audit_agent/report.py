"""Finding validation, de-duplication and the JSON output contract.

Output contract (assignment section 5)::

    {"repository": "<url>", "violations": [ {rule_id, category, severity, file,
      line, description, evidence, recommendation}, ... ]}

* severity is one of critical/high/medium/low;
* ``file``/``line`` are JSON ``null`` when not applicable;
* no secret value ever appears in the output (every text field is redacted).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .redact import redact_text

SEVERITIES = ("critical", "high", "medium", "low")
SEVERITY_ORDER = {s: i for i, s in enumerate(SEVERITIES)}

CATEGORIES = (
    "repository_hygiene",
    "dependency_management",
    "code_quality",
    "testing",
    "ci",
    "security",
    "configuration",
    "collaboration",
    "build_release",
)

# Accept a few natural aliases the LLM might produce and map them.
CATEGORY_ALIASES = {
    "hygiene": "repository_hygiene",
    "repository hygiene": "repository_hygiene",
    "dependencies": "dependency_management",
    "dependency management": "dependency_management",
    "code quality": "code_quality",
    "tests": "testing",
    "github_actions": "ci",
    "github actions": "ci",
    "github actions / ci": "ci",
    "ci/cd": "ci",
    "cicd": "ci",
    "config": "configuration",
    "build and release": "build_release",
    "build_and_release": "build_release",
    "release": "build_release",
}

RULE_ID_RE = re.compile(r"^[A-Z]{2,4}-\d{3}$")
REQUIRED_FIELDS = (
    "rule_id", "category", "severity", "file", "line", "description", "evidence", "recommendation",
)
_MAX_TEXT = 600


@dataclass
class Finding:
    rule_id: str
    category: str
    severity: str
    file: str | None
    line: int | None
    description: str
    evidence: str
    recommendation: str

    def key(self) -> tuple[str, str | None, int | None]:
        return (self.rule_id, self.file, self.line)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FindingError(ValueError):
    """Raised when a proposed finding does not satisfy the contract."""


def normalise_category(value: Any) -> str:
    if not isinstance(value, str):
        raise FindingError("category must be a string")
    v = value.strip().lower()
    v = CATEGORY_ALIASES.get(v, v).replace(" ", "_").replace("-", "_")
    v = CATEGORY_ALIASES.get(v, v)
    if v not in CATEGORIES:
        raise FindingError(f"category must be one of {list(CATEGORIES)}, got {value!r}")
    return v


def _clean_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FindingError(f"{field} must be a non-empty string")
    text = redact_text(" ".join(value.split()))
    if len(text) > _MAX_TEXT:
        text = text[: _MAX_TEXT - 3].rstrip() + "..."
    return text


def build_finding(
    data: dict[str, Any],
    *,
    file_exists=None,  # noqa: ANN001 - callable(path) -> bool
    line_count=None,  # noqa: ANN001 - callable(path) -> int | None
    known_secret_values: frozenset[str] = frozenset(),
) -> Finding:
    """Validate raw tool arguments from the LLM and return a clean Finding.

    Raises :class:`FindingError` with an actionable message on any problem so
    the agent can correct itself.
    """
    if not isinstance(data, dict):
        raise FindingError("finding must be an object")

    rule_id = str(data.get("rule_id", "")).strip().upper()
    if not RULE_ID_RE.match(rule_id):
        raise FindingError("rule_id must look like 'SEC-001' (2-4 capital letters, dash, 3 digits)")

    category = normalise_category(data.get("category"))

    severity = str(data.get("severity", "")).strip().lower()
    if severity not in SEVERITIES:
        raise FindingError(f"severity must be one of {list(SEVERITIES)}")

    file_val = data.get("file")
    if isinstance(file_val, str):
        file_val = file_val.strip().replace("\\", "/")
        while file_val.startswith("./"):
            file_val = file_val[2:]
        if file_val in ("", "null", "None", "."):
            file_val = None
        if file_val and (file_val.startswith("/") or ".." in Path(file_val).parts):
            raise FindingError("file must be a path relative to the repository root")
    elif file_val is not None:
        raise FindingError("file must be a string path or null")

    line_val = data.get("line")
    if isinstance(line_val, str) and line_val.strip().isdigit():
        line_val = int(line_val.strip())
    if line_val in ("", "null", "None", 0, "0") and not isinstance(line_val, bool):
        line_val = None
    if line_val is not None:
        if isinstance(line_val, bool) or not isinstance(line_val, int) or line_val < 1:
            raise FindingError("line must be a positive integer or null")
        if file_val is None:
            raise FindingError("line given without file; set line to null or provide the file")

    if file_val is not None and file_exists is not None and not file_exists(file_val):
        raise FindingError(
            f"file {file_val!r} is not a tracked file in the repository. If the problem is that a "
            "file is MISSING, set file to null."
        )
    if line_val is not None and line_count is not None:
        total = line_count(file_val)
        if total is not None and line_val > total:
            raise FindingError(f"line {line_val} is beyond the end of {file_val} ({total} lines)")

    finding = Finding(
        rule_id=rule_id,
        category=category,
        severity=severity,
        file=file_val,
        line=line_val,
        description=_clean_text(data.get("description"), "description"),
        evidence=_clean_text(data.get("evidence"), "evidence"),
        recommendation=_clean_text(data.get("recommendation"), "recommendation"),
    )
    # Defence in depth: never echo a raw secret value that the scanner saw.
    for secret in known_secret_values:
        if len(secret) >= 6:
            for field in ("description", "evidence", "recommendation"):
                text = getattr(finding, field)
                if secret in text:
                    setattr(finding, field, text.replace(secret, "[REDACTED]"))
    return finding


class FindingStore:
    """Holds findings recorded by the agent, with de-duplication."""

    def __init__(self) -> None:
        self._items: dict[str, Finding] = {}
        self._counter = 0

    def add(self, finding: Finding) -> str:
        for fid, existing in self._items.items():
            if existing.key() == finding.key():
                raise FindingError(
                    f"duplicate of {fid} (same rule_id, file and line); update by withdrawing {fid} first"
                )
            if (
                existing.rule_id == finding.rule_id
                and existing.file == finding.file
                and existing.file is not None
                and existing.description.lower() == finding.description.lower()
            ):
                raise FindingError(
                    f"duplicate of {fid}: the same problem in the same file is already recorded. "
                    "Report repeated instances of one problem as a single finding."
                )
        self._counter += 1
        fid = f"F{self._counter}"
        self._items[fid] = finding
        return fid

    def withdraw(self, fid: str) -> Finding:
        if fid not in self._items:
            raise FindingError(f"no finding with id {fid!r}; current ids: {list(self._items)}")
        return self._items.pop(fid)

    def items(self) -> list[tuple[str, Finding]]:
        return list(self._items.items())

    def __len__(self) -> int:
        return len(self._items)

    def summary(self) -> str:
        if not self._items:
            return "(no findings recorded)"
        return "\n".join(
            f"{fid}: [{f.severity}] {f.rule_id} {f.category} {f.file or '-'}:{f.line or '-'} - {f.description}"
            for fid, f in self._items.items()
        )


def build_report(repository: str, findings: list[Finding]) -> dict[str, Any]:
    ordered = sorted(
        findings,
        key=lambda f: (SEVERITY_ORDER[f.severity], f.category, f.file or "", f.line or 0, f.rule_id),
    )
    return {"repository": repository, "violations": [f.to_dict() for f in ordered]}


def validate_report(report: Any) -> list[str]:
    """Return a list of schema problems (empty list = valid)."""
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["report must be a JSON object"]
    if set(report) != {"repository", "violations"}:
        errors.append(f"top-level keys must be exactly repository and violations, got {sorted(report)}")
    if not isinstance(report.get("repository"), str):
        errors.append("repository must be a string")
    violations = report.get("violations")
    if not isinstance(violations, list):
        return errors + ["violations must be an array"]
    for i, v in enumerate(violations):
        if not isinstance(v, dict):
            errors.append(f"violations[{i}] must be an object")
            continue
        missing = [k for k in REQUIRED_FIELDS if k not in v]
        if missing:
            errors.append(f"violations[{i}] missing fields {missing}")
        if v.get("severity") not in SEVERITIES:
            errors.append(f"violations[{i}].severity invalid: {v.get('severity')!r}")
        if v.get("file") is not None and not isinstance(v.get("file"), str):
            errors.append(f"violations[{i}].file must be string or null")
        line = v.get("line")
        if line is not None and (isinstance(line, bool) or not isinstance(line, int)):
            errors.append(f"violations[{i}].line must be integer or null")
        for k in ("rule_id", "category", "description", "evidence", "recommendation"):
            if k in v and not isinstance(v[k], str):
                errors.append(f"violations[{i}].{k} must be a string")
    return errors


def write_report(path: str | os.PathLike[str], report: dict[str, Any]) -> Path:
    """Validate and atomically write the report as UTF-8 JSON."""
    problems = validate_report(report)
    if problems:
        raise ValueError("refusing to write invalid report: " + "; ".join(problems))
    out = Path(path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".report-", suffix=".json", dir=str(out.parent or Path(".")))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, out)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return out
