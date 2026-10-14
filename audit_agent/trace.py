"""JSONL trace of every LLM decision and tool execution.

The trace lets an evaluator verify that the LLM genuinely directed the
investigation (which tool it chose, with what arguments, and why).  All
content is redacted before it is written.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from .redact import redact_obj

log = logging.getLogger("devops_audit")

_PREVIEW = 2000


class Tracer:
    def __init__(self, path: str | Path | None = None, verbose: bool = False) -> None:
        self.path = Path(path) if path else None
        self.verbose = verbose
        self._fh = None
        self._t0 = time.monotonic()
        self.events: list[dict[str, Any]] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", encoding="utf-8")

    def event(self, kind: str, **data: Any) -> None:
        record = {"t": round(time.monotonic() - self._t0, 3), "event": kind}
        record.update(redact_obj(data))
        for key in ("output", "text", "content"):
            val = record.get(key)
            if isinstance(val, str) and len(val) > _PREVIEW:
                record[key] = val[:_PREVIEW] + f"... [{len(val) - _PREVIEW} more chars]"
        self.events.append(record)
        if self._fh:
            self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        if self.verbose:
            self._echo(record)

    def _echo(self, record: dict[str, Any]) -> None:
        kind = record["event"]
        if kind == "tool_call":
            args = json.dumps(record.get("arguments", {}), ensure_ascii=False)[:200]
            print(f"[{record['t']:7.1f}s] tool  {record.get('name')}({args})", file=sys.stderr)
        elif kind == "llm_text" and record.get("text"):
            print(f"[{record['t']:7.1f}s] think {record['text'][:300]}", file=sys.stderr)
        elif kind in ("finding_recorded", "finding_rejected", "finding_withdrawn"):
            print(f"[{record['t']:7.1f}s] {kind}: {record.get('summary') or record.get('error')}",
                  file=sys.stderr)
        elif kind in ("gate", "warning", "error", "start", "end"):
            msg = record.get("message") or record.get("repository") or ""
            print(f"[{record['t']:7.1f}s] {kind}: {msg}", file=sys.stderr)

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
