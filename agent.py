#!/usr/bin/env python3
"""DevOps repository auditing agent - command-line entry point.

Usage:
    python agent.py --repo <github_repository_url> --output report.json

Exit codes:
    0  inspection completed, report written (possibly with an empty violations array)
    2  invalid input (bad arguments / not a GitHub repository URL)
    3  repository could not be cloned (missing, private, network, timeout)
    4  LLM configuration or provider error
    5  report could not be written
    1  unexpected internal error
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

# Make "tools" and "audit_agent" importable no matter where agent.py is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_agent import __version__  # noqa: E402
from audit_agent.llm import LLMClient, LLMError, create_client  # noqa: E402
from audit_agent.loop import AuditAgent  # noqa: E402
from audit_agent.redact import redact_text  # noqa: E402
from audit_agent.repo import (  # noqa: E402
    CloneError,
    InvalidRepositoryURL,
    RepoRef,
    cleanup,
    clone_repository,
    parse_github_url,
    working_tree_fingerprint,
)
from audit_agent.report import build_report, write_report  # noqa: E402
from audit_agent.trace import Tracer  # noqa: E402
from tools import RepoTools  # noqa: E402

EXIT_OK, EXIT_INTERNAL, EXIT_INPUT, EXIT_CLONE, EXIT_LLM, EXIT_OUTPUT = 0, 1, 2, 3, 4, 5


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
    except ImportError:
        return
    load_dotenv(Path.cwd() / ".env")
    load_dotenv(Path(__file__).resolve().parent / ".env")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent.py",
        description="AI agent that audits a public GitHub repository for DevOps best-practice violations "
                    "and writes a JSON report.",
    )
    p.add_argument("--repo", required=True, help="Public GitHub repository URL, e.g. https://github.com/o/r")
    p.add_argument("--output", required=True, help="Path of the JSON report to write, e.g. report.json")
    p.add_argument("--provider", default=None,
                   help="LLM provider: anthropic, openai, gemini, groq, openrouter, ollama "
                        "(default: LLM_PROVIDER env or the first provider with an API key)")
    p.add_argument("--model", default=None, help="Model name (default: LLM_MODEL env or provider default)")
    p.add_argument("--max-steps", type=int, default=int(os.getenv("AGENT_MAX_STEPS", "60")),
                   help="Maximum LLM reasoning steps (default 60)")
    p.add_argument("--time-budget", type=float, default=float(os.getenv("AGENT_TIME_BUDGET", "1500")),
                   help="Soft wall-clock budget in seconds for the investigation (default 1500)")
    p.add_argument("--clone-timeout", type=int, default=300, help="git clone timeout in seconds (default 300)")
    p.add_argument("--trace", default=None,
                   help="Where to write the JSONL tool-call trace (default: <output>.trace.jsonl; 'none' to disable)")
    p.add_argument("--verbose", "-v", action="store_true", help="Print the agent's reasoning and tool calls")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(
    argv: list[str] | None = None,
    *,
    llm_factory: Callable[[str | None, str | None], LLMClient] | None = None,
    cloner: Callable[[RepoRef, int], Path] | None = None,
) -> int:
    """Run the CLI. ``llm_factory``/``cloner`` are injection points for tests."""
    _load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    def fail(code: int, msg: str) -> int:
        print(f"error: {redact_text(msg)}", file=sys.stderr)
        return code

    if args.max_steps < 5:
        return fail(EXIT_INPUT, "--max-steps must be at least 5")
    try:
        ref = parse_github_url(args.repo)
    except InvalidRepositoryURL as exc:
        return fail(EXIT_INPUT, str(exc))
    out_path = Path(args.output)
    if out_path.exists() and out_path.is_dir():
        return fail(EXIT_INPUT, f"--output {out_path} is a directory")

    # Configure the LLM before cloning so configuration errors fail fast.
    try:
        llm = (llm_factory or create_client)(args.provider, args.model)
    except LLMError as exc:
        return fail(EXIT_LLM, str(exc))

    trace_path = None
    if (args.trace or "").lower() != "none":
        trace_path = Path(args.trace) if args.trace else out_path.with_suffix(out_path.suffix + ".trace.jsonl")
    tracer = Tracer(trace_path, verbose=args.verbose)

    clone_dir: Path | None = None
    try:
        print(f"Cloning {ref.url} ...", file=sys.stderr)
        try:
            clone_dir = (cloner or clone_repository)(ref, args.clone_timeout)
        except CloneError as exc:
            tracer.event("error", message=str(exc))
            return fail(EXIT_CLONE, str(exc))

        before = working_tree_fingerprint(clone_dir)
        tools = RepoTools(clone_dir)
        agent = AuditAgent(llm, tools, tracer, max_steps=args.max_steps, repo_url=ref.url,
                           time_budget_s=args.time_budget)
        print(f"Auditing with {getattr(llm, 'provider', '?')}/{getattr(llm, 'model', '?')} "
              f"(max {args.max_steps} steps) ...", file=sys.stderr)
        try:
            result = agent.run()
        except LLMError as exc:
            tracer.event("error", message=str(exc))
            return fail(EXIT_LLM, f"LLM provider error, audit aborted: {exc}")

        after = working_tree_fingerprint(clone_dir)
        if before != after:
            tracer.event("warning", message="working tree changed during audit (should never happen)")
        if not result.finished:
            print(f"warning: agent stopped before calling finish ({result.stop_reason}); "
                  "writing the findings recorded so far.", file=sys.stderr)

        report = build_report(ref.url, result.findings)
        try:
            write_report(out_path, report)
        except (OSError, ValueError) as exc:
            return fail(EXIT_OUTPUT, f"could not write report: {exc}")
        print(f"Wrote {out_path} with {len(report['violations'])} violation(s) "
              f"[{result.steps} steps, {result.tool_calls} tool calls"
              + (f", trace: {trace_path}" if trace_path else "") + "]", file=sys.stderr)
        return EXIT_OK
    except KeyboardInterrupt:
        return fail(EXIT_INTERNAL, "interrupted")
    except Exception as exc:  # last-resort guard: useful message, non-zero exit
        tracer.event("error", message=f"{type(exc).__name__}: {exc}")
        return fail(EXIT_INTERNAL, f"unexpected error: {type(exc).__name__}: {exc}")
    finally:
        tracer.close()
        if clone_dir is not None and cloner is None:
            cleanup(clone_dir.parent)


if __name__ == "__main__":
    sys.exit(main())
