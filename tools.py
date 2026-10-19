"""Inspection tools available to the auditing agent.

Required tools (assignment section 3): ``list_files``, ``read_file``,
``search_code``, ``git_log``.  Optional tools add deeper, *static* analysis:
``git_status``, ``git_show_commit``, ``inspect_workflows``,
``check_dependencies``, ``detect_project_tooling``, ``scan_secrets``,
``run_ruff`` and ``code_metrics``.

Safety guarantees:
* every tool is read-only; paths are confined to the cloned repository;
* repository code is never imported or executed (Python is parsed with
  ``ast``, YAML with ``yaml.safe_load``, ruff only parses source);
* every result is passed through secret redaction before being returned;
* outputs are size-limited so one tool call cannot flood the LLM context.

Each tool returns a plain-text observation for the LLM.  The agent records
which files/lines a tool has shown it (``observed``) so that findings can be
checked against evidence the agent actually saw.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import os
import re
import shutil
import subprocess
import time

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10: use the tomli backport if present
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from audit_agent.redact import find_secrets, redact_text

MAX_OUTPUT_CHARS = 12_000
MAX_FILE_BYTES = 1_000_000
MAX_READ_LINES = 400
SUBPROCESS_TIMEOUT = 60

TEXT_EXTENSIONS_BINARY_HINT = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz", ".tar", ".tgz", ".7z", ".whl",
    ".jar", ".so", ".dll", ".exe", ".bin", ".pyc", ".pyo", ".class", ".o", ".a", ".dylib", ".woff",
    ".woff2", ".ttf", ".eot", ".mp3", ".mp4", ".mov", ".avi", ".pkl", ".pickle", ".h5", ".pt",
    ".onnx", ".parquet", ".sqlite", ".db", ".npy", ".npz", ".xlsx", ".docx", ".pptx",
}

# Tracked paths that usually should not be in version control.
GENERATED_PATTERNS = [
    ("python bytecode", re.compile(r"(^|/)__pycache__/|\.py[co]$")),
    ("virtual environment", re.compile(r"(^|/)(\.?venv|env|virtualenv)/(bin|lib|Lib|Scripts|pyvenv\.cfg)")),
    ("node_modules", re.compile(r"(^|/)node_modules/")),
    ("OS metadata", re.compile(r"(^|/)(\.DS_Store|Thumbs\.db|desktop\.ini)$")),
    ("editor swap/backup", re.compile(r"(\.swp|\.swo|~|\.bak|\.orig|\.tmp|\.temp)$")),
    ("log file", re.compile(r"\.log$")),
    ("build output", re.compile(r"(^|/)(dist|build|target|out)/|\.egg-info/")),
    ("coverage/test cache", re.compile(r"(^|/)(\.coverage$|htmlcov/|\.pytest_cache/|\.tox/|\.mypy_cache/|"
                                       r"\.ruff_cache/|coverage\.xml$|\.nyc_output/)")),
    ("IDE settings", re.compile(r"(^|/)(\.idea/|\.vscode/|[^/]+\.code-workspace$|[^/]+\.iml$)")),
    ("local env/secret file", re.compile(r"(^|/)(\.env(\.[A-Za-z0-9_-]+)?|[^/]*\.pem|[^/]*\.key|id_rsa[^/]*|"
                                         r"[^/]*\.p12|[^/]*\.pfx|\.npmrc|\.pypirc|credentials(\.json)?|"
                                         r"secrets?\.(json|ya?ml|toml))$")),
    ("compiled binary", re.compile(r"\.(so|dll|exe|o|a|dylib|class|jar)$")),
]
_ENV_TEMPLATE = re.compile(r"(^|/)\.env\.(example|sample|template|dist|defaults?)$", re.I)


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [output truncated: {len(text) - limit} more characters]"


def _git_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({"GIT_OPTIONAL_LOCKS": "0", "GIT_PAGER": "cat", "GIT_TERMINAL_PROMPT": "0",
                "GIT_CONFIG_NOSYSTEM": "1", "LC_ALL": "C"})
    return env


class ToolError(Exception):
    """Recoverable tool error; the message is returned to the LLM."""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass
class RepoTools:
    """All inspection tools bound to one cloned repository."""

    root: Path
    observed: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    secret_values: set[str] = field(default_factory=set)
    _tracked: list[str] | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if not self.root.is_dir():
            raise ToolError(f"repository root {self.root} does not exist")

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _git(self, *args: str, timeout: int = SUBPROCESS_TIMEOUT) -> str:
        try:
            proc = subprocess.run(
                ["git", "-c", "core.quotepath=off", *args], cwd=self.root, capture_output=True,
                text=True, errors="replace", timeout=timeout, env=_git_env(), check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"git {args[0]} timed out after {timeout}s") from exc
        except OSError as exc:
            raise ToolError(f"git unavailable: {exc}") from exc
        if proc.returncode != 0:
            raise ToolError(f"git {args[0]} failed: {proc.stderr.strip()[:300]}")
        return proc.stdout

    def tracked_files(self) -> list[str]:
        if self._tracked is None:
            try:
                out = self._git("ls-files", "-z")
                files = [f for f in out.split("\0") if f]
            except ToolError:
                files = []
                for dirpath, dirnames, filenames in os.walk(self.root):
                    dirnames[:] = [d for d in dirnames if d != ".git"]
                    for fn in filenames:
                        files.append(str(Path(dirpath, fn).relative_to(self.root).as_posix()))
            self._tracked = sorted(files)
        return self._tracked

    def file_exists(self, rel: str) -> bool:
        if not isinstance(rel, str) or not rel.strip():
            return False
        return _norm_rel(rel) in set(self.tracked_files())

    def _safe_path(self, rel: str, must_exist: bool = True) -> Path | None:
        if not isinstance(rel, str) or not rel.strip():
            raise ToolError("path must be a non-empty string relative to the repository root")
        rel = rel.strip().replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        if rel.startswith("/") or ".." in PurePosixPath(rel).parts or rel.startswith(".git/") or rel == ".git":
            raise ToolError("path must stay inside the repository (no absolute paths, '..' or .git/)")
        p = (self.root / rel)
        try:
            resolved = p.resolve()
        except OSError as exc:
            raise ToolError(f"cannot resolve path: {exc}") from exc
        if self.root not in resolved.parents and resolved != self.root:
            raise ToolError("path resolves outside the repository (symlink escape blocked)")
        if must_exist and not p.exists():
            raise ToolError(f"{rel} does not exist in the repository")
        return p

    def line_count(self, rel: str) -> int | None:
        try:
            p = self._safe_path(rel)
        except ToolError:
            return None
        if p is None or not p.is_file():
            return None
        try:
            with p.open("rb") as fh:
                data = fh.read(MAX_FILE_BYTES * 5)
            return data.count(b"\n") + (0 if data.endswith(b"\n") or not data else 1)
        except OSError:
            return None

    @staticmethod
    def _is_binary(p: Path) -> bool:
        if p.suffix.lower() in TEXT_EXTENSIONS_BINARY_HINT:
            return True
        try:
            with p.open("rb") as fh:
                chunk = fh.read(8192)
            return b"\0" in chunk
        except OSError:
            return True

    def _read_text(self, rel: str) -> str | None:
        p = self.root / rel
        try:
            if not p.is_file() or p.is_symlink() and self.root not in p.resolve().parents:
                return None
            if p.stat().st_size > MAX_FILE_BYTES or self._is_binary(p):
                return None
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def _text_files(self, glob: str | None = None) -> list[str]:
        files = self.tracked_files()
        if glob:
            files = [f for f in files if _glob_match(f, glob)]
        return files

    def _observe(self, rel: str, lines: list[int] | range | None = None) -> None:
        bucket = self.observed[rel]
        if lines is not None:
            bucket.update(lines)

    def _remember_secrets(self, text: str) -> None:
        for hit in find_secrets(text):
            if not hit.placeholder and len(hit.value) >= 6:
                self.secret_values.add(hit.value)

    # ------------------------------------------------------------------ #
    # REQUIRED TOOLS
    # ------------------------------------------------------------------ #
    def list_files(self, path: str = "", pattern: str | None = None, max_entries: int = 300) -> str:
        """List tracked files (git ls-files) with sizes and hygiene flags."""
        max_entries = max(20, min(int(max_entries or 300), 1000))
        files = self.tracked_files()
        prefix = path.strip().strip("/") if path else ""
        if prefix:
            self._safe_path(prefix, must_exist=False)
            files = [f for f in files if f == prefix or f.startswith(prefix + "/")]
        if pattern:
            files = [f for f in files if _glob_match(f, pattern)]
        total = len(files)
        lines = [f"{total} tracked file(s)" + (f" under '{prefix}'" if prefix else "")
                 + (f" matching '{pattern}'" if pattern else "") + ":"]

        flagged: list[str] = []
        for f in files:
            for label, rx in GENERATED_PATTERNS:
                if label == "local env/secret file" and _ENV_TEMPLATE.search(f):
                    continue
                if rx.search(f):
                    flagged.append(f"{f}  <- {label}")
                    self._observe(f, [])  # existence is the evidence for hygiene findings
                    break

        if total > max_entries:
            # Summarise by directory to keep output manageable.
            dirs = Counter(str(PurePosixPath(f).parent) for f in files)
            lines.append(f"(too many to list; showing directory summary and first {max_entries // 2} files)")
            for d, n in sorted(dirs.items())[:150]:
                lines.append(f"  {d}/  ({n} files)")
            lines.append("First files:")
            files = files[: max_entries // 2]
        for f in files:
            p = self.root / f
            try:
                size = p.stat().st_size if not p.is_symlink() else 0
            except OSError:
                size = -1
            suffix = " [symlink]" if p.is_symlink() else ""
            lines.append(f"  {f}  ({_human(size)}){suffix}")
        if flagged:
            lines.append("\nPaths that are commonly generated/local/secret (verify before reporting):")
            lines.extend(f"  {x}" for x in flagged[:80])
            if len(flagged) > 80:
                lines.append(f"  ... and {len(flagged) - 80} more")
        large = [(f, (self.root / f).stat().st_size) for f in self.tracked_files()
                 if (self.root / f).is_file() and not (self.root / f).is_symlink()
                 and (self.root / f).stat().st_size > 5_000_000]
        if large:
            for f, _ in large:
                self._observe(f, [])
            lines.append("\nLarge tracked files (>5 MB):")
            lines.extend(f"  {f} ({_human(s)})" for f, s in large[:20])
        return "\n".join(lines)

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a text file with line numbers (secrets redacted)."""
        p = self._safe_path(path)
        assert p is not None
        rel = _norm_rel(path)
        if p.is_dir():
            raise ToolError(f"{rel} is a directory; use list_files(path='{rel}')")
        size = p.stat().st_size
        if self._is_binary(p):
            self._observe(rel, [])
            return f"{rel}: binary file ({_human(size)}), not displayed."
        if size > MAX_FILE_BYTES * 5:
            raise ToolError(f"{rel} is too large ({_human(size)}) to read; use search_code instead")
        text = p.read_text(encoding="utf-8", errors="replace")
        self._remember_secrets(text)
        all_lines = text.splitlines()
        total = len(all_lines)
        start = max(1, int(start_line or 1))
        end = int(end_line) if end_line else start + MAX_READ_LINES - 1
        end = min(end, total, start + MAX_READ_LINES - 1)
        if total == 0:
            self._observe(rel, [])
            return f"{rel}: empty file (0 lines)."
        if start > total:
            raise ToolError(f"start_line {start} is beyond end of file ({total} lines)")
        self._observe(rel, range(start, end + 1))
        body = "\n".join(f"{i:>5}| {redact_text(all_lines[i - 1])[:400]}" for i in range(start, end + 1))
        header = f"{rel} (lines {start}-{end} of {total})"
        more = f"\n... {total - end} more lines; call read_file with start_line={end + 1}" if end < total else ""
        return _truncate(f"{header}\n{body}{more}")

    def search_code(self, pattern: str, path_glob: str | None = None, case_insensitive: bool = False,
                    max_results: int = 60) -> str:
        """Regex search across tracked text files."""
        if not pattern:
            raise ToolError("pattern must be a non-empty regular expression")
        try:
            rx = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from exc
        max_results = max(1, min(int(max_results or 60), 200))
        results: list[str] = []
        files_hit: set[str] = set()
        deadline = time.monotonic() + 30
        truncated = False
        for rel in self._text_files(path_glob):
            if time.monotonic() > deadline:
                truncated = True
                break
            text = self._read_text(rel)
            if text is None:
                continue
            for no, line in enumerate(text.splitlines(), start=1):
                if rx.search(line[:2000]):
                    if len(results) >= max_results:
                        truncated = True
                        break
                    self._remember_secrets(line)
                    self._observe(rel, [no])
                    files_hit.add(rel)
                    results.append(f"{rel}:{no}: {redact_text(line.strip())[:300]}")
            if truncated:
                break
        if not results:
            return f"No matches for /{pattern}/" + (f" in files matching '{path_glob}'" if path_glob else "")
        head = f"{len(results)} match(es) in {len(files_hit)} file(s)" + (" (truncated)" if truncated else "")
        return _truncate(head + ":\n" + "\n".join(results))

    def git_log(self, max_count: int = 25, path: str | None = None, show_files: bool = False) -> str:
        """Recent commit history (shallow clone: recent commits only)."""
        max_count = max(1, min(int(max_count or 25), 100))
        args = ["log", f"-n{max_count}", "--date=short", "--format=%h|%ad|%an|%s"]
        if show_files:
            args.append("--name-status")
        if path:
            self._safe_path(path, must_exist=False)
            args += ["--", path]
        out = self._git(*args)
        if not out.strip():
            return "No commits found" + (f" touching {path}" if path else "")
        lines: list[str] = []
        for line in out.splitlines():
            if "|" in line and line.count("|") >= 3 and not line.startswith(("A\t", "M\t", "D\t", "R")):
                sha, date, author, subject = line.split("|", 3)
                lines.append(f"{sha} {date} {author}: {subject}")
            elif line.strip():
                lines.append(f"    {line}")
        try:
            shallow = self._git("rev-parse", "--is-shallow-repository").strip() == "true"
            total = self._git("rev-list", "--count", "HEAD").strip()
        except ToolError:
            shallow, total = False, "?"
        shallow_note = "; shallow clone, older history not fetched" if shallow else ""
        note = f"\n(commits available locally: {total}{shallow_note})"
        return _truncate(redact_text("\n".join(lines)) + note)

    # ------------------------------------------------------------------ #
    # OPTIONAL TOOLS
    # ------------------------------------------------------------------ #
    def git_status(self) -> str:
        """Branch, HEAD commit and working-tree state of the clone."""
        branch = self._git("rev-parse", "--abbrev-ref", "HEAD").strip()
        head = self._git("log", "-1", "--format=%h %ad %s", "--date=short").strip()
        status = self._git("status", "--porcelain").strip()
        return (f"default branch (checked out): {branch}\nHEAD: {redact_text(head)}\n"
                f"working tree: {'clean (unchanged)' if not status else 'MODIFIED:' + chr(10) + status}")

    def git_show_commit(self, commit: str, path: str | None = None) -> str:
        """Show one commit's metadata and (redacted) diff."""
        if not re.fullmatch(r"[0-9a-fA-F]{4,40}", commit or ""):
            raise ToolError("commit must be a hexadecimal commit hash (4-40 chars)")
        args = ["show", "--stat", "--patch", "--format=%h %ad %an%n%s%n%b", "--date=short", commit]
        if path:
            self._safe_path(path, must_exist=False)
            args += ["--", path]
        out = self._git(*args)
        self._remember_secrets(out)
        return _truncate(redact_text(out), 8000)

    def inspect_workflows(self) -> str:
        """Parse GitHub Actions workflows and highlight risky patterns."""
        wf_files = [f for f in self.tracked_files()
                    if re.match(r"^\.github/workflows/[^/]+\.ya?ml$", f)]
        other_ci = [f for f in self.tracked_files() if _is_other_ci(f)]
        if not wf_files:
            msg = "No GitHub Actions workflows found in .github/workflows/."
            if other_ci:
                msg += " Other CI configuration present (acceptable alternative): " + ", ".join(other_ci)
            else:
                msg += " No other CI configuration files detected either."
            return msg
        try:
            default_branch = self._git("rev-parse", "--abbrev-ref", "HEAD").strip()
        except ToolError:
            default_branch = "unknown"
        sections = [f"Repository default branch: {default_branch}"]
        any_tests = False
        for rel in wf_files:
            text = self._read_text(rel)
            if text is None:
                sections.append(f"\n== {rel}: unreadable")
                continue
            self._remember_secrets(text)
            self._observe(rel, [])
            summary, has_tests = _summarise_workflow(rel, text, default_branch)
            any_tests = any_tests or has_tests
            sections.append(summary)
        if not any_tests:
            sections.append("\nNOTE: no workflow step appears to run tests "
                            "(pytest/unittest/tox/nox/npm test/go test/make test/etc.).")
        if other_ci:
            sections.append("Other CI configuration also present: " + ", ".join(other_ci))
        return _truncate(redact_text("\n".join(sections)), 16000)

    def check_dependencies(self) -> str:
        """Analyse dependency manifests, version constraints and lockfiles."""
        files = set(self.tracked_files())
        report: list[str] = []
        specs: dict[str, list[tuple[str, str]]] = defaultdict(list)

        def note(pkg: str, spec: str, src: str) -> None:
            specs[_norm_pkg(pkg)].append((src, spec or "(any)"))

        req_files = sorted(f for f in files if re.search(r"(^|/)requirements[^/]*\.(txt|in)$", f)
                           or f.endswith("constraints.txt"))
        for rel in req_files:
            text = self._read_text(rel) or ""
            self._observe(rel, [])
            stats = Counter()
            unpinned: list[str] = []
            issues: list[str] = []
            for no, raw in enumerate(text.splitlines(), start=1):
                line = raw.split(" #")[0].strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(("-r ", "--requirement", "-c ", "--constraint")):
                    stats["includes"] += 1
                    continue
                if line.startswith(("--index-url", "-i ", "--extra-index-url", "--trusted-host")):
                    if "http://" in line or "--trusted-host" in line:
                        issues.append(f"line {no}: insecure package index option: {redact_text(line)[:120]}")
                    continue
                if line.startswith(("-e ", "--editable")) or "://" in line or line.startswith("git+"):
                    if ("git+" in line or "://" in line) and not re.search(r"@[0-9a-f]{7,40}\b", line):
                        issues.append(f"line {no}: VCS/URL dependency not pinned to a commit: "
                                      f"{redact_text(line)[:120]}")
                    stats["url/editable"] += 1
                    continue
                name, spec = _split_requirement(line)
                kind = _classify_spec(spec)
                stats[kind] += 1
                note(name, spec, rel)
                if kind in ("unpinned", "lower-bound only"):
                    unpinned.append(f"{name}{spec} (line {no})")
                self._observe(rel, [no])
            report.append(f"{rel}: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())) or "empty")
            if unpinned:
                report.append(f"  unpinned/lower-bound-only: {', '.join(unpinned[:25])}"
                              + (" ..." if len(unpinned) > 25 else ""))
            report.extend(f"  {i}" for i in issues)

        for rel in sorted(f for f in files if f.endswith("pyproject.toml")):
            text = self._read_text(rel) or ""
            self._observe(rel, [])
            try:
                data = _toml_loads(text)
            except _TomlUnavailable:
                report.append(f"{rel}: could not be parsed on this Python (no TOML parser); not checked")
                continue
            except ValueError as exc:
                report.append(f"{rel}: INVALID TOML ({exc})")
                continue
            project = data.get("project", {}) or {}
            deps = list(project.get("dependencies", []) or [])
            for items in (project.get("optional-dependencies") or {}).values():
                deps.extend(items or [])
            kinds = Counter()
            loose: list[str] = []
            for d in deps:
                name, spec = _split_requirement(d)
                kind = _classify_spec(spec)
                kinds[kind] += 1
                note(name, spec, rel)
                if kind == "unpinned":
                    loose.append(name)
            poetry = (data.get("tool", {}) or {}).get("poetry", {}) or {}
            pdeps = dict(poetry.get("dependencies", {}) or {})
            for grp in (poetry.get("group", {}) or {}).values():
                pdeps.update((grp or {}).get("dependencies", {}) or {})
            for name, val in pdeps.items():
                if name.lower() == "python":
                    continue
                spec = val if isinstance(val, str) else (val.get("version", "") if isinstance(val, dict) else "")
                kind = "unpinned" if spec in ("*", "") else _classify_spec(spec)
                kinds[kind] += 1
                note(name, spec, rel)
                if kind == "unpinned":
                    loose.append(name)
            tools = sorted((data.get("tool", {}) or {}).keys())
            build = (data.get("build-system", {}) or {}).get("build-backend")
            report.append(f"{rel}: project deps {dict(kinds) or 'none'}; build-backend={build}; "
                          f"[tool.*] sections: {', '.join(tools) or 'none'}"
                          + ("; uses Poetry" if poetry else ""))
            if loose:
                report.append(f"  unconstrained: {', '.join(loose[:25])}")
            if project.get("requires-python") is None and project:
                report.append("  note: no requires-python declared")

        for rel in sorted(f for f in files if f.endswith("setup.py")):
            text = self._read_text(rel) or ""
            m = re.search(r"install_requires\s*=\s*\[(.*?)\]", text, re.S)
            if m:
                reqs = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
                for r in reqs:
                    n, s = _split_requirement(r)
                    note(n, s, rel)
                report.append(f"{rel}: install_requires has {len(reqs)} entries "
                              f"({sum(1 for r in reqs if _classify_spec(_split_requirement(r)[1]) == 'unpinned')} "
                              "unconstrained)")
            self._observe(rel, [])

        for rel in sorted(f for f in files if f.endswith("Pipfile")):
            report.append(f"{rel}: Pipfile present; Pipfile.lock "
                          + ("present" if rel + ".lock" in files else "MISSING"))

        for rel in sorted(f for f in files if f.endswith("package.json") and "node_modules/" not in f):
            text = self._read_text(rel) or ""
            self._observe(rel, [])
            try:
                pkg = json.loads(text)
            except json.JSONDecodeError as exc:
                report.append(f"{rel}: INVALID JSON ({exc})")
                continue
            base = str(PurePosixPath(rel).parent)
            base = "" if base == "." else base + "/"
            locks = [lf for lf in ("package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb", "bun.lock",
                                   "npm-shrinkwrap.json") if base + lf in files]
            loose = []
            all_deps: dict[str, str] = {}
            for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
                for name, ver in (pkg.get(section) or {}).items():
                    all_deps[name] = str(ver)
                    if str(ver).strip() in ("*", "latest", "", "x") or str(ver).startswith(("http", "git")):
                        loose.append(f"{name}@{ver}")
            report.append(f"{rel}: {len(all_deps)} deps; lockfile: {', '.join(locks) or 'NONE'}; "
                          f"scripts: {', '.join(sorted((pkg.get('scripts') or {}).keys())) or 'none'}")
            if loose:
                report.append(f"  wildcard/latest/URL versions: {', '.join(loose[:20])}")
            if "package-lock.json" in locks:
                lock_txt = self._read_text(base + "package-lock.json") or ""
                try:
                    lock = json.loads(lock_txt)
                    root_pkg = (lock.get("packages") or {}).get("", {})
                    locked = {**(root_pkg.get("dependencies") or {}), **(root_pkg.get("devDependencies") or {})}
                    if locked:
                        missing = [d for d in all_deps if d not in locked]
                        if missing:
                            report.append(f"  package-lock.json OUT OF SYNC: not locked: {', '.join(missing[:15])}")
                except json.JSONDecodeError:
                    report.append("  package-lock.json is not valid JSON")

        lock_names = {"poetry.lock", "uv.lock", "Pipfile.lock", "pdm.lock", "requirements.lock",
                      "go.sum", "Cargo.lock", "Gemfile.lock", "composer.lock"}
        present_locks = sorted(f for f in files if PurePosixPath(f).name in lock_names)
        for rel in sorted(f for f in files if PurePosixPath(f).name == "poetry.lock"):
            pyproj = str(PurePosixPath(rel).parent / "pyproject.toml").removeprefix("./")
            text = self._read_text(rel) or ""
            locked = {_norm_pkg(n) for n in re.findall(r'^name\s*=\s*"([^"]+)"', text, re.M)}
            declared = [p for p, srcs in specs.items() if any(s == pyproj for s, _ in srcs)]
            stale = [p for p in declared if p not in locked]
            if stale and locked:
                report.append(f"{rel}: OUT OF SYNC with {pyproj}; not locked: {', '.join(stale[:15])}")
        for rel in sorted(f for f in files if PurePosixPath(f).name == "go.mod"):
            report.append(f"{rel}: go.sum " + ("present" if str(PurePosixPath(rel).with_name("go.sum"))
                                                in files else "MISSING"))
        for rel in sorted(f for f in files if PurePosixPath(f).name == "Cargo.toml"):
            report.append(f"{rel}: Cargo.lock " + ("present" if any(f.endswith("Cargo.lock") for f in files)
                                                   else "absent (fine for libraries, expected for binaries)"))
        for rel in sorted(f for f in files if PurePosixPath(f).name in ("environment.yml", "environment.yaml")):
            report.append(f"{rel}: conda environment file present")

        conflicts = []
        for pkg, entries in specs.items():
            distinct = {s for _, s in entries if s != "(any)"}
            if len(distinct) > 1 and len({src for src, _ in entries}) > 1:
                conflicts.append(f"{pkg}: " + "; ".join(f"{src} -> {s}" for src, s in entries))
        if not report:
            report.append("No dependency manifests found (requirements*.txt, pyproject.toml, setup.py, Pipfile, "
                          "package.json, go.mod, Cargo.toml ...).")
        if conflicts:
            report.append("\nDIFFERENT SPECIFICATIONS FOR THE SAME PACKAGE ACROSS FILES:")
            report.extend(f"  {c}" for c in conflicts[:30])
        report.append("\nLockfiles present: " + (", ".join(present_locks) or "none"))
        report.append("Hint: libraries usually use ranges without a lockfile; deployable applications "
                      "should pin or lock. Judge using detect_project_tooling / README context.")
        return _truncate(redact_text("\n".join(report)))

    def detect_project_tooling(self) -> str:
        """Detect languages, linters, formatters, test setup, CI and packaging (acceptable alternatives)."""
        files = self.tracked_files()
        fset = set(files)
        names = {PurePosixPath(f).name for f in files}
        lang = Counter(PurePosixPath(f).suffix.lower() for f in files if PurePosixPath(f).suffix)
        out = ["Top file extensions: " + ", ".join(f"{e}={n}" for e, n in lang.most_common(8))]

        def has(*cands: str) -> list[str]:
            return [c for c in cands if c in fset or c in names]

        pyproject_tools: set[str] = set()
        for rel in (f for f in files if f.endswith("pyproject.toml")):
            try:
                pyproject_tools |= set((_toml_loads(self._read_text(rel) or "").get("tool") or {}).keys())
            except ValueError:
                pass
        cfg_text = ""
        for rel in ("setup.cfg", "tox.ini"):
            if rel in fset:
                cfg_text += self._read_text(rel) or ""

        linters = []
        for tool in ("ruff", "black", "isort", "flake8", "pylint", "mypy", "pyright", "autopep8", "yapf"):
            if tool in pyproject_tools:
                linters.append(f"{tool} (pyproject [tool.{tool}])")
        linters += [f"{x} file" for x in has("ruff.toml", ".ruff.toml", ".flake8", ".pylintrc", "pylintrc",
                                               "mypy.ini", ".mypy.ini", ".isort.cfg", ".editorconfig",
                                               ".eslintrc", ".eslintrc.js", ".eslintrc.json", ".eslintrc.cjs",
                                               ".eslintrc.yml", "eslint.config.js", "eslint.config.mjs",
                                               ".prettierrc", ".prettierrc.json", "prettier.config.js",
                                               "biome.json", ".golangci.yml", ".golangci.yaml", "rustfmt.toml",
                                               ".rubocop.yml", ".stylelintrc")]
        for sec in ("flake8", "pylint", "mypy", "isort", "pycodestyle"):
            if f"[{sec}" in cfg_text:
                linters.append(f"{sec} (setup.cfg/tox.ini)")
        precommit = has(".pre-commit-config.yaml", ".pre-commit-config.yml")
        if precommit:
            txt = self._read_text(precommit[0]) or ""
            hooks = re.findall(r"id:\s*([\w\-]+)", txt)
            linters.append(f"pre-commit hooks: {', '.join(hooks[:15]) or '(none listed)'}")
        out.append("Lint/format tooling: " + ("; ".join(linters) if linters else "NONE detected"))

        test_files = [f for f in files if re.search(
            r"(^|/)(tests?/|test_[^/]+\.py$|[^/]+_test\.(py|go)$|[^/]+\.(test|spec)\.[jt]sx?$|__tests__/)", f)]
        test_cfg = [x for x in ("pytest.ini", "tox.ini", "noxfile.py", "conftest.py", "jest.config.js",
                                "jest.config.ts", "vitest.config.ts", "karma.conf.js", "phpunit.xml")
                    if x in names]
        if "pytest" in pyproject_tools or "[tool:pytest]" in cfg_text:
            test_cfg.append("pytest config")
        out.append(f"Test files: {len(test_files)}" + (f" e.g. {', '.join(test_files[:6])}" if test_files else "")
                   + f"; test config: {', '.join(test_cfg) or 'none'}")

        ci = [f for f in files if f.startswith(".github/workflows/")] + [f for f in files if _is_other_ci(f)]
        out.append("CI configuration: " + (", ".join(ci) if ci else "NONE detected"))
        build = has("Dockerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yaml", "Makefile",
                    "setup.py", "setup.cfg", "pyproject.toml", "package.json", "go.mod", "Cargo.toml",
                    "pom.xml", "build.gradle", "Procfile", ".dockerignore")
        out.append("Build/packaging files: " + (", ".join(build) or "none"))
        if "Makefile" in fset:
            targets = re.findall(r"^([A-Za-z0-9_\-]+):", self._read_text("Makefile") or "", re.M)
            out.append("Makefile targets: " + ", ".join(targets[:20]))
        docs = has("README.md", "README.rst", "README.txt", "README", "readme.md", "CONTRIBUTING.md",
                   "LICENSE", "LICENSE.md", "CHANGELOG.md", "CODEOWNERS", ".github/CODEOWNERS",
                   "docs/CODEOWNERS", ".github/pull_request_template.md", ".github/PULL_REQUEST_TEMPLATE.md",
                   "SECURITY.md", ".github/dependabot.yml", ".github/dependabot.yaml", "renovate.json")
        out.append("Docs/collaboration files: " + (", ".join(docs) or "none"))
        out.append(".gitignore: " + ("present" if ".gitignore" in fset else "MISSING")
                   + ("; .dockerignore present" if ".dockerignore" in fset else ""))
        env_examples = [f for f in files if _ENV_TEMPLATE.search(f)]
        out.append("Env templates: " + (", ".join(env_examples) or "none"))
        app_markers = [f for f in files if PurePosixPath(f).name in
                       ("app.py", "main.py", "manage.py", "wsgi.py", "asgi.py", "server.py", "Dockerfile",
                        "Procfile", "docker-compose.yml", "server.js", "index.js")]
        out.append("Application entry-point markers: " + (", ".join(app_markers[:10]) or "none (may be a library)"))
        return "\n".join(out)

    def scan_secrets(self, path_glob: str | None = None) -> str:
        """Scan tracked files for credentials. Values are never shown."""
        hits: list[str] = []          # source/config files: likely real
        illustrative: list[str] = []  # tests, docs, examples: usually fake/example values
        placeholders = 0
        for rel in self._text_files(path_glob):
            text = self._read_text(rel)
            if text is None or not _SECRET_PREFILTER.search(text):  # fast skip for large repos
                continue
            example_file = _is_test_path(rel) or _is_doc_path(rel)
            for no, line in enumerate(text.splitlines(), start=1):
                if not _SECRET_PREFILTER.search(line):
                    continue
                for h in find_secrets(line):
                    if h.placeholder or _is_example_value(h.kind, h.value):
                        placeholders += 1
                        continue
                    self.secret_values.add(h.value)
                    self._observe(rel, [no])
                    entry = f"{rel}:{no}: {h.kind} -> {redact_text(line.strip())[:200]}"
                    (illustrative if example_file else hits).append(entry)
        secret_rx = dict(GENERATED_PATTERNS)["local env/secret file"]
        sensitive = [f for f in self.tracked_files()
                     if secret_rx.search(f) and not _ENV_TEMPLATE.search(f)]
        out = [f"{len(hits)} likely real credential(s) in source/config files (values redacted):"] + hits[:60]
        if len(hits) > 60:
            out.append(f"... {len(hits) - 60} more")
        if illustrative:
            out.append(f"\n{len(illustrative)} credential-looking value(s) in tests/docs/examples. These are USUALLY "
                       "illustrative fixtures and NOT violations; report one only if it is clearly a real, "
                       "live credential (e.g. a real provider key format in an example config):")
            out.extend(f"  {h}" for h in illustrative[:15])
            if len(illustrative) > 15:
                out.append(f"  ... {len(illustrative) - 15} more")
        if placeholders:
            out.append(f"({placeholders} placeholder-looking values ignored, e.g. 'changeme', '${{VAR}}', 'your-key')")
        if sensitive:
            out.append("Tracked files that typically hold secrets/local config: " + ", ".join(sensitive[:30]))
        history = self._deleted_file_secrets()
        if history:
            out.append("SECRETS STILL IN GIT HISTORY (file deleted later, but the value remains in history; report "
                       "with file=null since the file is no longer tracked):")
            out.extend(f"  {h}" for h in history[:20])
        out.append("Tip: if a credential was removed in a later commit, check git_log(show_files=true) "
                   "and git_show_commit.")
        return _truncate("\n".join(out))

    def _deleted_file_secrets(self) -> list[str]:
        """Scan files that were deleted in history (still present in older commits)."""
        try:
            log = self._git("log", "--diff-filter=D", "--name-only", "--format=@%h", "-n", "200")
        except ToolError:
            return []
        results: list[str] = []
        sha = ""
        for line in log.splitlines():
            if line.startswith("@"):
                sha = line[1:].strip()
                continue
            path = line.strip()
            if not path or not sha:
                continue
            try:
                content = self._git("show", f"{sha}^:{path}")
            except ToolError:  # shallow clone boundary or binary
                continue
            for no, text_line in enumerate(content.splitlines(), start=1):
                for h in find_secrets(text_line):
                    if not h.placeholder:
                        self.secret_values.add(h.value)
                        results.append(f"{path}:{no} {h.kind} (deleted in commit {sha}; "
                                       f"see git_show_commit('{sha}'))")
            if len(results) > 40:
                break
        return results

    def run_ruff(self, path: str = "") -> str:
        """Static lint with ruff (parses code, never executes it)."""
        exe = shutil.which("ruff")
        if exe is None:
            return "ruff is not installed in the auditing environment; check unavailable (do not infer results)."
        py_files = [f for f in self.tracked_files() if f.endswith(".py")]
        if not py_files:
            return "No Python files tracked; ruff not applicable."
        target = "."
        if path:
            self._safe_path(path)
            target = path
        cmd = [exe, "check", "--isolated", "--no-cache", "--exit-zero", "--output-format", "json",
               "--select", "E9,F63,F7,F82,F401,F811,F841,E711,E712,E722,B006,B008,S105,S106,S108,S113,S501",
               "--target-version", "py311", target]
        try:
            proc = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=120, check=False,
                                  env={**os.environ, "RUFF_CACHE_DIR": os.devnull})
        except subprocess.TimeoutExpired:
            return "ruff timed out; check unavailable."
        try:
            items = json.loads(proc.stdout or "[]")
        except json.JSONDecodeError:
            # Older ruff without some rule codes: retry with core rules only.
            cmd[cmd.index("--select") + 1] = "E9,F63,F7,F82,F401,F841,E722"
            proc = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=120, check=False)
            try:
                items = json.loads(proc.stdout or "[]")
            except json.JSONDecodeError:
                return f"ruff failed: {redact_text(proc.stderr)[:300]}"
        if not items:
            return f"ruff: no issues in {len(py_files)} Python file(s) for the selected rules."
        by_code = Counter(i.get("code") or "syntax" for i in items)
        out = [f"ruff found {len(items)} issue(s) in {len({i['filename'] for i in items})} file(s). "
               "By rule: " + ", ".join(f"{c}={n}" for c, n in by_code.most_common())]
        out.append("Severe = syntax errors (E9/'syntax'), undefined names (F821/F822), bare except (E722), "
                   "S105/S106 hardcoded passwords, S501 verify=False. F401/F841 are minor.")
        for i in sorted(items, key=lambda x: (x.get("code") or "", x["filename"]))[:60]:
            rel = os.path.relpath(i["filename"], self.root) if os.path.isabs(i["filename"]) else i["filename"]
            row = (i.get("location") or {}).get("row")
            self._observe(rel, [row] if row else [])
            out.append(f"{rel}:{row}: {i.get('code') or 'syntax'} {redact_text(i.get('message', ''))[:150]}")
        return _truncate("\n".join(out))

    def code_metrics(self) -> str:
        """AST-based maintainability metrics for Python files (no execution)."""
        py_files = [f for f in self.tracked_files() if f.endswith(".py")]
        if not py_files:
            return "No Python files tracked."
        long_funcs, big_files, bare_except, syntax_errors = [], [], [], []
        debug, unresolved = [], []
        total_lines = 0
        local_modules = _local_python_modules(py_files)
        local_tops = {m.split(".")[0] for m in local_modules}
        for rel in py_files:
            text = self._read_text(rel)
            if text is None:
                continue
            n = text.count("\n") + 1
            total_lines += n
            if n > 800:
                big_files.append(f"{rel} ({n} lines)")
            try:
                tree = ast.parse(text)
            except SyntaxError as exc:
                syntax_errors.append(f"{rel}:{exc.lineno}: {exc.msg}")
                self._observe(rel, [exc.lineno or 1])
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    length = (getattr(node, "end_lineno", node.lineno) or node.lineno) - node.lineno + 1
                    if length > 100:
                        long_funcs.append((length, f"{rel}:{node.lineno} {node.name}() {length} lines"))
                elif isinstance(node, ast.ExceptHandler) and node.type is None:
                    bare_except.append(f"{rel}:{node.lineno}")
                    self._observe(rel, [node.lineno])
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "breakpoint":
                    debug.append(f"{rel}:{node.lineno} breakpoint()")
                    self._observe(rel, [node.lineno])
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = ([node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
                             else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                    for mod in names:
                        if mod.split(".")[0] in local_tops and mod not in local_modules:
                            unresolved.append(f"{rel}:{node.lineno} imports '{mod}' (no such module in the repo)")
                            self._observe(rel, [node.lineno])
        out = [f"{len(py_files)} Python files, {total_lines} lines."]
        out.append("Syntax errors: " + (", ".join(syntax_errors[:20]) or "none"))
        out.append("Functions > 100 lines: " + (", ".join(x for _, x in sorted(long_funcs, reverse=True)[:10])
                                                 or "none"))
        out.append("Files > 800 lines: " + (", ".join(big_files[:10]) or "none"))
        out.append(f"Bare 'except:' clauses: {len(bare_except)}" + (f" ({', '.join(bare_except[:15])})"
                                                                    if bare_except else ""))
        out.append("Leftover breakpoint() calls: " + (", ".join(debug[:10]) or "none"))
        out.append("Imports of the project's own modules that do not exist (code/tests cannot run): "
                   + ("; ".join(unresolved[:15]) or "none"))
        return _truncate("\n".join(out))

    # ------------------------------------------------------------------ #
    # registry
    # ------------------------------------------------------------------ #
    def specs(self) -> list[ToolSpec]:
        S = ToolSpec
        return [
            S("list_files",
              "List tracked files (git ls-files) with sizes. Flags paths that look generated, local or secret "
              "(e.g. __pycache__, .env, .DS_Store, venv, IDE settings). Use 'path' to drill into a directory and "
              "'pattern' (glob, e.g. '*.yml' or '**/Dockerfile') to filter.",
              _obj({"path": _str("Directory prefix relative to repo root ('' for all)."),
                    "pattern": _str("Optional glob filter, e.g. '*.py' or '.github/**'."),
                    "max_entries": _int("Maximum entries to list (20-1000).")}),
              self.list_files),
            S("read_file",
              "Read a text file with line numbers (max 400 lines per call; secrets are redacted). Use start_line/"
              "end_line for long files. Always read a file before reporting a violation in it.",
              _obj({"path": _str("File path relative to the repository root."),
                    "start_line": _int("First line to show (1-based)."),
                    "end_line": _int("Last line to show.")}, ["path"]),
              self.read_file),
            S("search_code",
              "Search tracked text files with a Python regular expression. Returns file:line: content. Use for "
              "suspicious patterns: hardcoded paths ('/home/|C:\\\\\\\\Users'), credentials, 'verify=False', "
              "'DEBUG = True', '|| true', TODO, etc.",
              _obj({"pattern": _str("Python regular expression."),
                    "path_glob": _str("Optional glob restricting files, e.g. '*.py' or 'src/**'."),
                    "case_insensitive": {"type": "boolean", "description": "Case-insensitive match."},
                    "max_results": _int("Maximum matches (1-200).")}, ["pattern"]),
              self.search_code),
            S("git_log",
              "Show recent commit history (hash, date, author, subject). show_files=true adds changed files per "
              "commit, useful to spot committed-then-deleted secrets or generated files.",
              _obj({"max_count": _int("Number of commits (1-100)."),
                    "path": _str("Only commits touching this path."),
                    "show_files": {"type": "boolean", "description": "Include changed files per commit."}}),
              self.git_log),
            S("git_status", "Show the checked-out default branch, HEAD commit and whether the clone is unmodified.",
              _obj({}), self.git_status),
            S("git_show_commit", "Show one commit's message and diff (secrets redacted).",
              _obj({"commit": _str("Commit hash from git_log."),
                    "path": _str("Optional path to limit the diff.")}, ["commit"]),
              self.git_show_commit),
            S("inspect_workflows",
              "Parse all GitHub Actions workflows: triggers, branch filters vs default branch, permissions, jobs, "
              "steps, action pinning, and risky patterns (pull_request_target with PR checkout, script injection, "
              "continue-on-error, '|| true', secrets echoed, deploy without test gate). Reports other CI systems.",
              _obj({}), self.inspect_workflows),
            S("check_dependencies",
              "Analyse dependency manifests (requirements*.txt, pyproject.toml, setup.py, Pipfile, package.json, "
              "go.mod, Cargo.toml): version constraints, conflicting specs across files, lockfile presence and "
              "lockfile/manifest drift.",
              _obj({}), self.check_dependencies),
            S("detect_project_tooling",
              "Detect languages, lint/format tools, test files/config, CI systems, build/packaging files, docs, "
              "collaboration files and env templates. Use it to recognise ACCEPTABLE ALTERNATIVES before "
              "reporting something as missing.",
              _obj({}), self.detect_project_tooling),
            S("scan_secrets",
              "Scan tracked files for real-looking credentials (API keys, tokens, private keys, passwords). "
              "Returns locations and secret type only; values are never shown. Placeholders are ignored.",
              _obj({"path_glob": _str("Optional glob restricting files.")}), self.scan_secrets),
            S("run_ruff",
              "Run the ruff linter statically (no code execution) with a curated rule set: syntax errors, "
              "undefined names, unused imports/variables, bare except, mutable defaults, hardcoded passwords, "
              "TLS verify disabled.",
              _obj({"path": _str("Optional file or directory to lint.")}), self.run_ruff),
            S("code_metrics",
              "AST-based Python checks: syntax errors, imports of project modules that do not exist (e.g. a test "
              "importing a missing module, so tests cannot pass), very long functions/files, bare except, "
              "leftover breakpoint() calls.",
              _obj({}), self.code_metrics),
        ]


# ---------------------------------------------------------------------- #
# workflow analysis
# ---------------------------------------------------------------------- #
class _LineLoader(yaml.SafeLoader):
    """SafeLoader that records the source line of every mapping as '__line__'."""


def _construct_mapping(loader: _LineLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping = yaml.SafeLoader.construct_mapping(loader, node, deep=deep)
    mapping["__line__"] = node.start_mark.line + 1
    return mapping


_LineLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)

_UNTRUSTED_CTX = re.compile(
    r"\$\{\{\s*github\.(?:event\.(?:issue\.(?:title|body)|pull_request\.(?:title|body|head\.ref|head\.label)|"
    r"comment\.body|review\.body|review_comment\.body|pages\.[^}]*page_name|commits[^}]*(?:message|author)|"
    r"head_commit\.(?:message|author)|discussion\.(?:title|body)|workflow_run\.head_branch)|head_ref)\s*\}\}"
)
_TEST_CMD = re.compile(r"\b(pytest|py\.test|unittest|tox|nox|npm (run )?test|yarn test|pnpm test|go test|"
                       r"cargo test|mvn (-\S+ )*test|gradle(w)? test|make test|jest|vitest|phpunit|rspec)\b")
_MASKING = re.compile(r"(\|\|\s*(true|exit 0|:)\b|;\s*exit 0\s*$|set \+e|--exit-zero|-\s*-exit-zero)", re.M)
_MUTABLE_REFS = {"main", "master", "latest", "dev", "develop", "head", "trunk"}
_P_IF = r"\bif\s*:"
_P_REF = r"\bref\s*:"
_P_ECHO_SECRET = r"(echo|printf|cat)\b.*secrets\."  # noqa: S105 - regex, not a password


def _summarise_workflow(rel: str, text: str, default_branch: str) -> tuple[str, bool]:
    try:
        data = yaml.load(text, Loader=_LineLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark is not None else ""
        flags, has_tests = _scan_workflow_lines(text)
        out = [f"\n== {rel}: INVALID YAML{where}: {' '.join(str(exc).split())[:250]}",
               "  ! GitHub Actions cannot load this workflow, so NONE of its jobs run "
               "(CI silently disabled).",
               "Line-based fallback scan (the file could not be parsed):"]
        out += [f"  ! {f}" for f in flags] or ["  (no risky patterns found by line scan)"]
        return "\n".join(out), has_tests
    if not isinstance(data, dict):
        return f"\n== {rel}: not a mapping", False
    lines = [f"\n== {rel}  (name: {data.get('name', '-')})"]
    flags: list[str] = []
    src = text.splitlines()

    def find(start: int | None, pattern: str, default: int | None = None) -> int | None:
        """First line >= start matching pattern: points findings at the exact offending line."""
        rx = re.compile(pattern)
        for no in range(max(1, start or 1), len(src) + 1):
            if not src[no - 1].lstrip().startswith("#") and rx.search(src[no - 1]):
                return no
        return default if default is not None else start
    triggers = data.get("on", data.get(True))  # PyYAML parses bare 'on' as True
    trig_names: list[str] = []
    if isinstance(triggers, str):
        trig_names = [triggers]
    elif isinstance(triggers, list):
        trig_names = [str(t) for t in triggers]
    elif isinstance(triggers, dict):
        trig_names = [str(k) for k in triggers if k != "__line__"]
    lines.append(f"triggers: {', '.join(trig_names) or 'NONE'}")
    if isinstance(triggers, dict):
        for ev in ("push", "pull_request", "pull_request_target"):
            cfg = triggers.get(ev)
            if isinstance(cfg, dict):
                br = cfg.get("branches")
                if br is not None:
                    br_list = br if isinstance(br, list) else [br]
                    bl = find(cfg.get("__line__"), r"\bbranches\s*:")
                    lines.append(f"  {ev}.branches: {br_list} (line {bl})")
                    if default_branch not in br_list and not any(any(c in str(b) for c in "*?[") for b in br_list):
                        flags.append(f"{ev} branch filter {br_list} does not include the default branch "
                                     f"'{default_branch}' (line {bl}) -> workflow may never run "
                                     "for the main line of development")
                for key in ("paths", "paths-ignore", "branches-ignore", "tags"):
                    if cfg.get(key) is not None:
                        lines.append(f"  {ev}.{key}: {cfg.get(key)}")
                if cfg.get("branches-ignore") and default_branch in (cfg.get("branches-ignore") or []):
                    flags.append(f"{ev} ignores the default branch '{default_branch}'")
    if trig_names and set(trig_names) <= {"workflow_dispatch", "schedule"}:
        flags.append("only manual/scheduled triggers: CI does not run on push or pull_request")
    if "pull_request_target" in trig_names:
        flags.append("uses pull_request_target (runs with base-repo secrets/write token) - check whether it "
                     "checks out or runs PR head code")
    if "workflow_run" in trig_names:
        flags.append("uses workflow_run - check whether it downloads/executes artifacts from untrusted runs")

    perms = data.get("permissions")
    lines.append(f"top-level permissions: {_fmt_perm(perms)}")
    if perms == "write-all":
        flags.append("top-level permissions: write-all (over-privileged GITHUB_TOKEN)")
    elif isinstance(perms, dict):
        writes = [k for k, v in perms.items() if k != "__line__" and v == "write"]
        if writes:
            lines.append(f"  write scopes: {writes}")
    elif perms is None:
        lines.append("  (no top-level permissions block; token scope falls back to repository/org default "
                     "settings, which cannot be verified from files)")

    jobs = data.get("jobs") or {}
    has_tests = False
    test_jobs: set[str] = set()
    for job_id, job in jobs.items():
        if job_id == "__line__" or not isinstance(job, dict):
            continue
        jl = find(1, rf"^\s*{re.escape(str(job_id))}\s*:", job.get("__line__"))
        needs = job.get("needs")
        needs_l = needs if isinstance(needs, list) else ([needs] if needs else [])
        lines.append(f"job '{job_id}' (line {jl}): runs-on={job.get('runs-on')}, needs={needs_l or '-'}, "
                     f"if={job.get('if', '-')}, permissions={_fmt_perm(job.get('permissions'))}"
                     + (f", uses reusable workflow {job.get('uses')}" if job.get("uses") else ""))
        if job.get("permissions") == "write-all":
            flags.append(f"job '{job_id}' permissions: write-all (line {find(jl, 'write-all', jl)})")
        if job.get("continue-on-error") in (True, "true"):
            flags.append(f"job '{job_id}' has continue-on-error: true (line {find(jl, 'continue-on-error', jl)}) "
                         "- failures never fail the run")
        if str(job.get("if", "")).strip() in ("false", "${{ false }}", "0"):
            flags.append(f"job '{job_id}' is disabled with if: false (line {find(jl, _P_IF, jl)})")
        if job.get("environment"):
            lines.append(f"  environment: {job.get('environment')}")
        steps = job.get("steps") or []
        job_runs_tests = False
        is_deploy = bool(re.search(r"deploy|release|publish|push|upload", str(job_id), re.I))
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            sl = step.get("__line__")
            name = step.get("name") or step.get("uses") or (str(step.get("run", ""))[:40])
            uses = step.get("uses")
            run = str(step.get("run", "") or "")
            desc = f"  step {idx + 1} (line {sl}): {name}"
            if uses:
                desc += f" | uses {uses}"
                ref = str(uses).split("@", 1)[1] if "@" in str(uses) else ""
                owner = str(uses).split("/", 1)[0]
                if not str(uses).startswith(("./", "docker://")):
                    if not ref:
                        flags.append(f"action '{uses}' has no version ref (line {find(sl, re.escape(str(uses)), sl)})")
                    elif ref.lower() in _MUTABLE_REFS:
                        flags.append(f"action '{uses}' pinned to mutable branch '{ref}' "
                                     f"(line {find(sl, re.escape(str(uses)), sl)})")
                    elif owner not in ("actions", "github") and not re.fullmatch(r"[0-9a-f]{40}", ref):
                        lines.append(f"    (third-party action pinned to tag '{ref}', not a commit SHA)")
                if "checkout" in str(uses):
                    w = step.get("with") or {}
                    ref_in = str(w.get("ref", ""))
                    if "pull_request.head" in ref_in or "head_ref" in ref_in or "github.event.pull_request" in ref_in:
                        if "pull_request_target" in trig_names or "workflow_run" in trig_names:
                            event = "pull_request_target" if "pull_request_target" in trig_names else "workflow_run"
                            flags.append(f"checks out untrusted PR head ({ref_in}) in a privileged {event} "
                                         f"workflow (line {find(sl, _P_REF, sl)}) - critical 'pwn request' risk")
                if re.search(r"deploy|release|publish|pypi|docker/build-push|aws-actions|azure/|"
                             r"google-github-actions|heroku|vercel|netlify|gh-pages", str(uses), re.I):
                    is_deploy = True
            if run:
                first = " ".join(run.strip().split())[:160]
                desc += f" | run: {first}"
                if _TEST_CMD.search(run):
                    job_runs_tests = True
                m = _MASKING.search(run)
                if m:
                    flags.append(f"step '{name}' masks failures with '{m.group(0).strip()}' "
                                 f"(line {find(sl, re.escape(m.group(0).strip()), sl)})")
                if _UNTRUSTED_CTX.search(run):
                    flags.append(f"step '{name}' interpolates untrusted github.event data directly into a shell "
                                 f"script (line {find(sl, _UNTRUSTED_CTX.pattern, sl)}) - script injection")
                if re.search(r"(echo|printf|cat)\b[^\n]*\$\{\{\s*secrets\.", run):
                    flags.append(f"step '{name}' echoes a secret into the log "
                                 f"(line {find(sl, _P_ECHO_SECRET, sl)})")
                if re.search(r"\b(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(ba)?sh\b", run):
                    flags.append(f"step '{name}' pipes a downloaded script to a shell "
                                 f"(line {find(sl, '(curl|wget)', sl)})")
                if re.search(r"(deploy|publish|twine upload|docker push|kubectl apply|helm (upgrade|install)|"
                             r"terraform apply|npm publish|gh release|git push)", run, re.I):
                    is_deploy = True
                if re.search(r"git push\s+(-f|--force)", run):
                    flags.append(f"step '{name}' force-pushes (line {find(sl, r'git push', sl)})")
                if re.search(r"gh pr (merge|review)[^\n]*(--auto|--approve|--admin)", run):
                    flags.append(f"step '{name}' auto-merges/approves pull requests (line {find(sl, r'gh pr', sl)}) "
                                 "- check gating")
            if step.get("continue-on-error") in (True, "true"):
                flags.append(f"step '{name}' has continue-on-error: true (line {find(sl, 'continue-on-error', sl)})")
            if str(step.get("if", "")).strip() in ("false", "${{ false }}", "0"):
                flags.append(f"step '{name}' is disabled with if: false (line {find(sl, _P_IF, sl)})")
            if step.get("env"):
                for k, v in step["env"].items():
                    if k != "__line__" and isinstance(v, str) and find_secrets(f'{k}="{v}"') \
                            and "${{" not in v:
                        flags.append(f"step '{name}' sets env {k} to a literal credential "
                                     f"(line {find(sl, re.escape(str(k)), sl)})")
            lines.append(desc)
        if job_runs_tests:
            has_tests = True
            test_jobs.add(job_id)
        if is_deploy:
            lines.append(f"  -> job '{job_id}' looks like a deploy/release job")
            if not needs_l and len([j for j in jobs if j != "__line__"]) > 1:
                flags.append(f"deploy/release job '{job_id}' has no 'needs:' dependency on test/build jobs "
                             f"(line {jl}) - can ship untested code")
            if trig_names and ("push" in trig_names or "pull_request" in trig_names):
                trig_cfg = triggers.get("push") if isinstance(triggers, dict) else None
                if not job.get("if") and not (isinstance(trig_cfg, dict) and (trig_cfg.get("branches")
                                                                             or trig_cfg.get("tags"))):
                    if "pull_request" in trig_names:
                        flags.append(f"deploy/release job '{job_id}' runs on pull_request events without an "
                                     f"'if' guard (line {jl})")
                    elif "push" in trig_names:
                        flags.append(f"deploy/release job '{job_id}' runs on push to ANY branch "
                                     f"(no branch/tag filter or if-guard) (line {jl})")
    if flags:
        lines.append("FLAGS (heuristic - confirm by reading the file):")
        lines.extend(f"  ! {f}" for f in flags)
    else:
        lines.append("FLAGS: none")
    lines.append(f"test-running jobs: {sorted(test_jobs) or 'none'}")
    return "\n".join(lines), has_tests


def _scan_workflow_lines(text: str) -> tuple[list[str], bool]:
    """Regex fallback for workflows that are not valid YAML."""
    flags: list[str] = []
    has_tests = False
    for no, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if s.startswith("#"):
            continue
        if _TEST_CMD.search(s):
            has_tests = True
        m = _MASKING.search(s)
        if m:
            flags.append(f"line {no}: masks failures with '{m.group(0).strip()}'")
        if _UNTRUSTED_CTX.search(s):
            flags.append(f"line {no}: untrusted github.event data interpolated (possible script injection)")
        if re.search(r"(echo|printf)\b.*\$\{\{\s*secrets\.", s):
            flags.append(f"line {no}: echoes a secret into the log")
        if re.search(r"continue-on-error:\s*true", s):
            flags.append(f"line {no}: continue-on-error: true")
        if re.search(r"permissions:\s*write-all", s):
            flags.append(f"line {no}: permissions: write-all")
        if "pull_request_target" in s:
            flags.append(f"line {no}: pull_request_target trigger")
        um = re.search(r"uses:\s*([^\s@#]+)@([^\s#]+)", s)
        if um and um.group(2).lower() in _MUTABLE_REFS:
            flags.append(f"line {no}: action {um.group(1)} pinned to mutable ref '{um.group(2)}'")
    return flags, has_tests


def _fmt_perm(perm: Any) -> str:
    if isinstance(perm, dict):
        return "{" + ", ".join(f"{k}: {v}" for k, v in perm.items() if k != "__line__") + "}"
    return str(perm) if perm is not None else "not set"


def _is_other_ci(f: str) -> bool:
    return bool(re.match(r"^(\.gitlab-ci\.yml|\.travis\.yml|Jenkinsfile|azure-pipelines\.ya?ml|"
                         r"bitbucket-pipelines\.yml|\.circleci/config\.ya?ml|\.drone\.yml|appveyor\.yml|"
                         r"\.buildkite/.*|cloudbuild\.ya?ml|buildspec\.ya?ml|\.woodpecker\.ya?ml)$", f))


# ---------------------------------------------------------------------- #
# small helpers
# ---------------------------------------------------------------------- #
def _local_python_modules(py_files: list[str]) -> set[str]:
    """Dotted names of every module/package defined by tracked .py files (src/ layout aware)."""
    mods: set[str] = set()
    for f in py_files:
        parts = list(PurePosixPath(f).with_suffix("").parts)
        for variant in (parts, parts[1:] if parts and parts[0] == "src" else None):
            if not variant:
                continue
            if variant[-1] == "__init__":
                variant = variant[:-1]
            for i in range(1, len(variant) + 1):
                mods.add(".".join(variant[:i]))
    return mods


_SECRET_PREFILTER = re.compile(
    r"(?i)key|secret|pass|pwd|token|credential|auth|://|-----BEGIN|AKIA|ASIA|gh[pousr]_|github_pat|glpat|xox|"
    r"AIza|sk-|sk_live|rk_live|eyJ|SG\.|hooks\.slack")

_DOC_RX = re.compile(r"(^|/)(docs?|docs?[_-]\w+|documentation|examples?|samples?|tutorials?|demos?)/|"
                     r"\.(md|rst|txt|adoc)$|(^|/)(README|CHANGELOG|HISTORY|CHANGES)[^/]*$", re.I)

# Values that are illustrative words rather than secrets, for the generic (key=value) detectors only.
_EXAMPLE_WORDS = {"pass", "password", "passwd", "secret", "test", "testing", "dev", "development", "key",
                  "token", "admin", "root", "user", "username", "foo", "bar", "baz", "qux", "example", "sample",
                  "demo", "default", "changeit", "letmein", "abc123", "123456", "1234", "12345678", "pwd",
                  "secret key", "test key", "dev key", "my secret", "mysecret", "supersecret", "s3cr3t", "hunter2"}
_VERSION_SPEC = re.compile(r"[\^~>=<!]*\s*v?\d+(\.[\dx*]+)*([-+.][\w.]+)?|[*x]|latest")
_GENERIC_SECRET_KINDS = {"hardcoded_credential", "env_credential", "connection_string_password"}


class _TomlUnavailable(ValueError):
    """No TOML parser on this Python (3.10 without the tomli backport)."""


def _toml_loads(text: str) -> dict[str, Any]:
    """Parse TOML; raises ValueError on invalid TOML, _TomlUnavailable if no parser exists."""
    if tomllib is None:
        raise _TomlUnavailable("TOML parsing unavailable on this Python (install 'tomli' or use Python 3.11+)")
    return tomllib.loads(text)  # TOMLDecodeError is a ValueError subclass


def _is_example_value(kind: str, value: str) -> bool:
    if kind not in _GENERIC_SECRET_KINDS:
        return False  # real credential FORMATS (AWS, GitHub, private keys...) are always reported
    v = value.strip().strip("'\"").lower()
    if _VERSION_SPEC.fullmatch(v):  # e.g. "pbkdf2-password": "1.2.1" in package.json is a dependency
        return True
    return v in _EXAMPLE_WORDS or (len(v) <= 6 and v.isalpha()) or v.startswith(("fake", "dummy", "example"))


def _is_doc_path(rel: str) -> bool:
    return bool(_DOC_RX.search(rel))


def _is_test_path(rel: str) -> bool:
    return bool(re.search(r"(^|/)(tests?|__tests__|spec|fixtures?|testdata)/|(^|/)test_[^/]*$|_test\.[a-z]+$|"
                          r"\.(test|spec)\.[jt]sx?$|(^|/)conftest\.py$", rel))


def _norm_rel(path: str) -> str:
    rel = path.strip().replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    return str(PurePosixPath(rel))


def _glob_match(path: str, pattern: str) -> bool:
    pattern = pattern.strip()
    if fnmatch.fnmatch(path, pattern):
        return True
    if "/" not in pattern and fnmatch.fnmatch(PurePosixPath(path).name, pattern):
        return True
    if pattern.startswith("**/") and fnmatch.fnmatch(path, pattern[3:]):
        return True
    if pattern.endswith("/**") and path.startswith(pattern[:-3].rstrip("/") + "/"):
        return True
    return False


def _human(n: int) -> str:
    if n < 0:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _norm_pkg(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


_REQ_SPLIT = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._\-]*)(\[[^\]]*\])?\s*(.*)$")


def _split_requirement(line: str) -> tuple[str, str]:
    line = line.split(";")[0].strip()
    m = _REQ_SPLIT.match(line)
    if not m:
        return line, ""
    return m.group(1), m.group(3).strip()


def _classify_spec(spec: str) -> str:
    s = spec.replace(" ", "")
    if not s or s in ("*",):
        return "unpinned"
    if s.startswith(("==", "===")) and "*" not in s:
        return "pinned"
    if "<" in s or s.startswith(("~=", "^", "~")) or "==" in s:
        return "bounded"
    if s.startswith((">=", ">")):
        return "lower-bound only"
    return "other"


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    # No "additionalProperties" key: some OpenAI-compatible providers (Gemini) reject it.
    return {"type": "object", "properties": props, "required": required or []}


def _str(desc: str) -> dict[str, str]:
    return {"type": "string", "description": desc}


def _int(desc: str) -> dict[str, str]:
    return {"type": "integer", "description": desc}
