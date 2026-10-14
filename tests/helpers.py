"""Shared test helpers: throwaway git repositories and a scripted fake LLM."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_agent.llm import LLMResponse, ToolCall  # noqa: E402

# A fake AWS key in the official "example" format, assembled at runtime so the
# test file itself contains no credential-looking literal.
FAKE_AWS_KEY = "AKIA" + "IOSFODNN7" + "EXAMPLQ"


def make_repo(files: dict[str, str | bytes], branch: str = "main", extra_commits: list[dict[str, Any]] | None = None
              ) -> Path:
    """Create a git repository in a temp dir with ``files`` committed."""
    root = Path(tempfile.mkdtemp(prefix="audit-test-"))
    _git(root, "init", "-q", "-b", branch)
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    # Byte-exact checkouts on every OS (Git for Windows enables autocrlf system-wide).
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "config", "core.safecrlf", "false")
    _write(root, files)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial commit", "--allow-empty")
    for commit in extra_commits or []:
        _write(root, commit.get("files", {}))
        for rel in commit.get("delete", []):
            (root / rel).unlink()
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", commit.get("message", "update"))
    return root


def _write(root: Path, files: dict[str, str | bytes]) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_bytes(content.encode("utf-8"))  # always LF, even on Windows


def _git(root: Path, *args: str) -> str:
    # Same environment the tools use (system git config ignored), so the repo
    # looks identical to the code under test on Windows, macOS and Linux.
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True, env=env).stdout


class ScriptedLLM:
    """Fake LLM that replays a fixed list of turns.

    Each turn is either an ``LLMResponse`` or a list of ``(tool_name, args)``
    tuples.  Every request it receives is stored for assertions.
    """

    provider = "fake"
    model = "scripted"

    def __init__(self, turns: list[Any]) -> None:
        self.turns = list(turns)
        self.requests: list[dict[str, Any]] = []
        self._n = 0

    def complete(self, system, messages, tools):  # noqa: ANN001,ANN201
        self.requests.append({"system": system, "messages": [dict(m) for m in messages],
                              "tools": [t["name"] for t in tools]})
        if not self.turns:
            return LLMResponse(text="nothing more to do")
        turn = self.turns.pop(0)
        if isinstance(turn, LLMResponse):
            return turn
        calls = []
        for name, args in turn:
            self._n += 1
            calls.append(ToolCall(id=f"call_{self._n}", name=name, arguments=args))
        return LLMResponse(text="reasoning about next step", tool_calls=calls)


FULL_COVERAGE = {
    "repository_hygiene": "checked README/.gitignore/tracked files",
    "dependency_management": "checked manifests",
    "code_quality": "ran ruff",
    "testing": "looked for tests",
    "ci": "inspected workflows",
    "security": "scanned secrets",
    "configuration": "searched for absolute paths",
    "collaboration": "checked PR automation",
    "build_release": "checked release jobs",
}
