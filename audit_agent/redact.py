"""Secret detection and redaction.

Every string that leaves a tool, goes to the LLM, is written to the trace or
ends up in the final report passes through :func:`redact_text`.  Secret values
are therefore never logged, never sent to the LLM provider and never written
to ``report.json``.  Detection results only carry the secret *type* and its
location.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Patterns for well-known credential formats.  Each entry: (type, regex,
# group index holding the secret value; 0 = whole match).
# --------------------------------------------------------------------------- #
_KNOWN_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"), 0),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), 0),
    (
        "aws_secret_access_key",
        re.compile(r"(?i)aws_?secret_?access_?key[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9/+=]{40})\b"),
        1,
    ),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})\b"), 0),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"), 0),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), 0),
    ("slack_webhook", re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/]+"), 0),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"), 0),
    ("anthropic_api_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"), 0),
    ("openai_api_key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}\b"), 0),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_live_[0-9A-Za-z]{16,}\b"), 0),
    ("sendgrid_key", re.compile(r"\bSG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\b"), 0),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), 0),
    (
        "connection_string_password",
        re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^:\s/@\"']+:([^@\s/\"']{3,})@"),
        1,
    ),
]

# key = "value" style assignments where the key name suggests a credential.
_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?P<key>[A-Za-z0-9_.\-]*?
        (?:password|passwd|pwd|secret|token|api[_\-]?key|apikey|access[_\-]?key|
           private[_\-]?key|client[_\-]?secret|auth[_\-]?key|credentials?)
     [A-Za-z0-9_.\-]*)
    ["']?\s*(?::=|=|:)\s*
    (?P<quote>["'])(?P<value>[^"'\s]{4,})(?P=quote)
    """
)

# Same, for .env / shell style files where the value is usually unquoted.
_ENV_ASSIGNMENT = re.compile(  # case-sensitive: environment variables are UPPER_CASE
    r"""(?x)
    ^\s*(?:export\s+)?
    (?P<key>[A-Za-z0-9_]*
        (?:PASSWORD|PASSWD|PWD|SECRET|TOKEN|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|CREDENTIALS?)
     [A-Za-z0-9_]*)
    \s*=\s*["']?(?P<value>[^"'\s#]{4,})["']?
    """
)

# Strong signals that a value is a placeholder, not a real secret.  Ambiguous
# words such as "test" are deliberately NOT included: a value like
# "test-secret-123" hardcoded in source is still reported.
_PLACEHOLDER_HINTS = (
    "changeme", "change_me", "change-me", "your_", "your-", "yourkey", "example", "placeholder",
    "dummy", "fake", "xxx", "***", "redacted", "<", ">", "${", "{{", "%(", "os.environ", "getenv",
    "env(", "process.env", "secrets.", "vault:", "replace_me", "replace-me", "insert", "....",
)


_GENERIC_KINDS = {"hardcoded_credential", "env_credential", "connection_string_password"}


@dataclass(frozen=True)
class SecretHit:
    """A detected secret. ``value`` is kept in memory only and never output."""

    kind: str
    start: int
    end: int
    value: str
    placeholder: bool


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for ch in value:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(value)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def looks_like_placeholder(value: str) -> bool:
    """Heuristic: is this an obvious placeholder rather than a real secret?"""
    low = value.lower()
    if any(h in low for h in _PLACEHOLDER_HINTS):
        return True
    if len(set(value)) <= 2:  # "xxxxxxxx", "00000000"
        return True
    if re.fullmatch(r"\{[^{}]*\}|%s|%\([^)]*\)s|\$[A-Za-z_][A-Za-z0-9_]*", value):  # format/template refs
        return True
    return False


def find_secrets(text: str) -> list[SecretHit]:
    """Return all secret-looking spans in ``text`` (non-overlapping)."""
    hits: list[SecretHit] = []

    def add(kind: str, m: re.Match[str], group: int | str) -> None:
        start, end = m.span(group)
        value = m.group(group)
        if any(not (end <= h.start or start >= h.end) for h in hits):
            return
        if kind in _GENERIC_KINDS:
            placeholder = looks_like_placeholder(value)
        else:  # a value matching a real credential FORMAT is reported even if it says EXAMPLE
            placeholder = len(set(value)) <= 2 or "${" in value or "{{" in value
        hits.append(SecretHit(kind, start, end, value, placeholder))

    for kind, pattern, group in _KNOWN_PATTERNS:
        for m in pattern.finditer(text):
            add(kind, m, group)

    for m in _ASSIGNMENT.finditer(text):
        value = m.group("value")
        # Skip code references such as  password = "{password}"  or  token = "TOKEN"
        if value.isupper() and "_" not in value and len(value) < 12:
            continue
        add("hardcoded_credential", m, "value")

    for line_match in re.finditer(r"[^\n]+", text):
        line = line_match.group(0)
        m = _ENV_ASSIGNMENT.match(line)
        if not m:
            continue
        start = line_match.start() + m.start("value")
        end = line_match.start() + m.end("value")
        value = m.group("value")
        if any(not (end <= h.start or start >= h.end) for h in hits):
            continue
        if value.startswith(("$", "%")):
            continue
        hits.append(SecretHit("env_credential", start, end, value, looks_like_placeholder(value)))

    hits.sort(key=lambda h: h.start)
    return hits


def redact_text(text: str) -> str:
    """Replace every detected secret value with a marker.

    Obvious placeholders ("changeme", "${API_KEY}", "your-key-here") are left
    visible so the agent can tell them apart from real leaked credentials.
    """
    if not text:
        return text
    hits = [h for h in find_secrets(text) if not h.placeholder]
    if not hits:
        return text
    out: list[str] = []
    pos = 0
    for h in hits:
        out.append(text[pos:h.start])
        if h.kind == "private_key":
            out.append("-----BEGIN [REDACTED] PRIVATE KEY-----")
        else:
            out.append(f"[REDACTED:{h.kind}]")
        pos = h.end
    out.append(text[pos:])
    return "".join(out)


def redact_obj(obj):  # noqa: ANN001,ANN201 - generic recursive helper
    """Recursively redact strings inside dicts/lists (used for traces)."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v) for v in obj]
    return obj


def high_entropy(value: str, threshold: float = 3.5) -> bool:
    return len(value) >= 16 and _entropy(value) >= threshold
