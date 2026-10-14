"""The explicit agent loop: Observe -> Reason -> Select Tool -> Execute -> Interpret -> Report.

No agent framework is used.  The loop:

1. sends the conversation and the tool schemas to the LLM;
2. the LLM *chooses* which inspection tool(s) to call next (dynamic selection);
3. the controller validates the arguments, dispatches the tool, redacts and
   truncates the observation and appends it to the conversation;
4. findings are recorded through the ``record_violation`` control tool and are
   validated against the repository (file must be tracked, line in range,
   evidence must have been observed, no duplicates, no secrets);
5. ``finish`` passes through two gates: a coverage gate (every category must be
   accounted for) and a self-review gate (the LLM re-examines each finding and
   withdraws unsupported ones) before the report is produced.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from tools import RepoTools, ToolError, ToolSpec

from . import prompts
from .llm import LLMClient, LLMError, LLMResponse
from .redact import redact_text
from .report import CATEGORIES, Finding, FindingError, FindingStore, build_finding
from .trace import Tracer

MAX_TOOL_CALLS_PER_TURN = 8
HISTORY_CHAR_BUDGET = 350_000  # compact old observations beyond this


@dataclass
class AgentResult:
    findings: list[Finding]
    steps: int
    tool_calls: int
    finished: bool
    stop_reason: str
    coverage: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class AuditAgent:
    def __init__(self, llm: LLMClient, tools: RepoTools, tracer: Tracer | None = None,
                 max_steps: int = 60, repo_url: str = "", time_budget_s: float | None = None) -> None:
        self.llm = llm
        self.tools = tools
        self.tracer = tracer or Tracer()
        self.max_steps = max_steps
        self.repo_url = repo_url
        self.time_budget_s = time_budget_s
        self.store = FindingStore()
        self.messages: list[dict[str, Any]] = []
        self._specs: dict[str, ToolSpec] = {s.name: s for s in tools.specs()}
        self._schemas = [s.schema() for s in tools.specs()] + prompts.CONTROL_TOOLS
        self._coverage_ok = False
        self._review_done = False
        self._finished = False
        self._coverage: dict[str, str] = {}
        self._summary = ""
        self.tool_calls = 0
        self.tools_used: list[str] = []
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    # ------------------------------------------------------------------ #
    def run(self) -> AgentResult:
        try:
            branch = self.tools._git("rev-parse", "--abbrev-ref", "HEAD").strip()
        except ToolError:
            branch = "unknown"
        task = prompts.INITIAL_TASK.format(url=self.repo_url or "(local)", branch=branch,
                                           n_files=len(self.tools.tracked_files()))
        self.messages.append({"role": "user", "content": task})
        self.tracer.event("start", repository=self.repo_url, provider=getattr(self.llm, "provider", "?"),
                          model=getattr(self.llm, "model", "?"), max_steps=self.max_steps)

        started = time.monotonic()
        nudges = 0
        stop_reason = "max_steps"
        step = 0
        for step in range(1, self.max_steps + 1):
            if self.time_budget_s and time.monotonic() - started > self.time_budget_s:
                stop_reason = "time_budget"
                self.tracer.event("warning", message="time budget exhausted; finalising with recorded findings")
                break
            self._compact_history()
            resp = self._call_llm()
            calls = resp.tool_calls[:MAX_TOOL_CALLS_PER_TURN]
            self.messages.append({
                "role": "assistant",
                "content": resp.text or None,
                "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments,
                                "provider_data": getattr(c, "provider_data", {}) or {}} for c in calls],
            })
            if resp.text:
                self.tracer.event("llm_text", step=step, text=resp.text)

            if not calls:
                nudges += 1
                if nudges > 2:
                    stop_reason = "llm_stopped"
                    self.tracer.event("warning", message="LLM stopped calling tools; finalising")
                    break
                self.messages.append({"role": "user", "content": prompts.NO_TOOL_NUDGE})
                continue
            nudges = 0

            for call in calls:
                output = self._dispatch(call.name, call.arguments, step)
                self.messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                      "content": output})

            if self._finished:
                stop_reason = "finished"
                break

            left = self.max_steps - step
            if left in (3, 1):
                self.messages.append({"role": "user", "content": prompts.BUDGET_WARNING.format(left=left)})

        findings = [f for _, f in self.store.items()]
        self.tracer.event("end", message=f"stop_reason={stop_reason}", steps=step, tool_calls=self.tool_calls,
                          findings=len(findings), tools_used=sorted(set(self.tools_used)), usage=self.usage)
        return AgentResult(findings=findings, steps=step, tool_calls=self.tool_calls,
                           finished=self._finished, stop_reason=stop_reason, coverage=self._coverage,
                           summary=self._summary, usage=self.usage)

    # ------------------------------------------------------------------ #
    def _call_llm(self) -> LLMResponse:
        system = prompts.SYSTEM_PROMPT
        t0 = time.monotonic()
        if self.tracer.verbose:
            print(f"  ... waiting for {getattr(self.llm, 'provider', 'LLM')} (step decision)",
                  file=sys.stderr, flush=True)
        try:
            resp = self.llm.complete(system, self.messages, self._schemas)
        except LLMError:
            raise
        except Exception as exc:  # defensive: never leak provider internals/keys
            raise LLMError(redact_text(f"{type(exc).__name__}: {exc}")[:500]) from exc
        self.tracer.event("llm_call", duration_s=round(time.monotonic() - t0, 2),
                          tool_calls=[c.name for c in resp.tool_calls], usage=resp.usage)
        for k, v in (resp.usage or {}).items():
            self.usage[k] = self.usage.get(k, 0) + int(v or 0)
        return resp

    def _dispatch(self, name: str, args: dict[str, Any], step: int) -> str:
        self.tool_calls += 1
        t0 = time.monotonic()
        args = args if isinstance(args, dict) else {}
        if "__invalid_arguments__" in args:
            out = "ERROR: tool arguments were not valid JSON. Retry with a valid JSON object."
        elif name == "record_violation":
            out = self._record(args)
        elif name == "withdraw_violation":
            out = self._withdraw(args)
        elif name == "finish":
            out = self._finish(args)
        elif name in self._specs:
            self.tools_used.append(name)
            out = self._run_tool(name, args)
        else:
            out = (f"ERROR: unknown tool '{name}'. Available tools: "
                   + ", ".join(list(self._specs) + ["record_violation", "withdraw_violation", "finish"]))
        out = redact_text(out)
        self.tracer.event("tool_call", step=step, name=name, arguments=args,
                          duration_s=round(time.monotonic() - t0, 3), output=out)
        return out

    def _run_tool(self, name: str, args: dict[str, Any]) -> str:
        spec = self._specs[name]
        allowed = set(spec.parameters.get("properties", {}))
        clean = {k: v for k, v in args.items() if k in allowed and v is not None}
        unknown = sorted(set(args) - allowed)
        missing = [r for r in spec.parameters.get("required", []) if r not in clean]
        if missing:
            return f"ERROR: missing required argument(s) {missing} for {name}."
        try:
            out = spec.func(**clean)
        except ToolError as exc:
            out = f"ERROR: {exc}"
        except (TypeError, ValueError) as exc:
            out = f"ERROR: bad arguments for {name}: {exc}"
        except Exception as exc:  # a tool bug must not kill the audit
            out = f"ERROR: {name} failed unexpectedly: {type(exc).__name__}: {exc}"
        if unknown:
            out = f"(ignored unknown argument(s): {unknown})\n" + out
        return out

    # ---------------- control tools ---------------- #
    def _record(self, args: dict[str, Any]) -> str:
        try:
            finding = build_finding(
                args,
                file_exists=self.tools.file_exists,
                line_count=self.tools.line_count,
                known_secret_values=frozenset(self.tools.secret_values),
            )
            if finding.file is not None and finding.file not in self.tools.observed:
                raise FindingError(
                    f"you have not inspected {finding.file} yet. Read it (read_file) or locate the evidence "
                    "with search_code before recording a finding about it."
                )
            fid = self.store.add(finding)
        except FindingError as exc:
            self.tracer.event("finding_rejected", error=str(exc), arguments=args)
            return f"REJECTED: {exc}"
        summary = f"{fid} [{finding.severity}] {finding.rule_id} {finding.file or '-'}:{finding.line or '-'}"
        self.tracer.event("finding_recorded", summary=summary, finding=finding.to_dict())
        # A new finding after the review gate needs review too.
        return f"Recorded {summary}. Total findings: {len(self.store)}."

    def _withdraw(self, args: dict[str, Any]) -> str:
        fid = str(args.get("finding_id", "")).strip().upper()
        try:
            f = self.store.withdraw(fid)
        except FindingError as exc:
            return f"ERROR: {exc}"
        self.tracer.event("finding_withdrawn", summary=f"{fid} {f.rule_id} {f.file}",
                          reason=args.get("reason", ""))
        return f"Withdrawn {fid}. Remaining findings: {len(self.store)}."

    def _finish(self, args: dict[str, Any]) -> str:
        coverage = args.get("coverage") if isinstance(args.get("coverage"), dict) else {}
        for k, v in coverage.items():
            if isinstance(v, str) and v.strip():
                self._coverage[str(k)] = redact_text(v.strip())[:400]
        if isinstance(args.get("summary"), str):
            self._summary = redact_text(args["summary"])[:2000]

        if not self._coverage_ok:
            missing = [c for c in CATEGORIES if not self._coverage.get(c)]
            if missing:
                self.tracer.event("gate", message=f"coverage gate: missing {missing}")
                return prompts.COVERAGE_GATE.format(missing=", ".join(missing))
            self._coverage_ok = True

        if not self._review_done:
            self._review_done = True
            self.tracer.event("gate", message=f"self-review gate over {len(self.store)} finding(s)")
            return prompts.REVIEW_GATE.format(findings=self.store.summary())

        self._finished = True
        self.tracer.event("gate", message="finish accepted")
        return f"Audit finished with {len(self.store)} finding(s)."

    # ---------------- context management ---------------- #
    def _compact_history(self) -> None:
        """Shrink old tool observations when the conversation grows too large."""
        total = sum(len(json.dumps(m, default=str)) for m in self.messages)
        if total <= HISTORY_CHAR_BUDGET:
            return
        # Keep the 12 most recent tool results intact.
        tool_idx = [i for i, m in enumerate(self.messages) if m["role"] == "tool"]
        for i in tool_idx[:-12]:
            content = self.messages[i]["content"]
            if len(content) > 800:
                self.messages[i]["content"] = content[:800] + "\n... [older observation compacted]"
            total = sum(len(json.dumps(m, default=str)) for m in self.messages)
            if total <= HISTORY_CHAR_BUDGET * 0.7:
                break
