"""GitHub URL validation and safe, read-only cloning.

Safety measures:
* only ``https://github.com/<owner>/<repo>`` URLs are accepted (no local paths,
  no other protocols, no option injection into git);
* the clone goes into a fresh temporary directory, never the caller's tree;
* git hooks, LFS smudge filters, credential prompts and submodules are disabled;
* the working tree is made read-only after cloning so no tool can modify it;
* nothing is ever pushed - the remote repository is never changed.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .redact import redact_text

_GITHUB_URL = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9._-]{1,100}?)"
    r"(?:\.git)?/?$"
)


class InvalidRepositoryURL(ValueError):
    """The supplied repository URL is not a public GitHub repository URL."""


class CloneError(RuntimeError):
    """Cloning the repository failed (not found, private, network, timeout)."""


@dataclass(frozen=True)
class RepoRef:
    url: str  # normalised https URL, no trailing .git
    owner: str
    name: str


def parse_github_url(url: str) -> RepoRef:
    if not isinstance(url, str) or not url.strip():
        raise InvalidRepositoryURL("Repository URL is empty.")
    candidate = url.strip()
    m = _GITHUB_URL.match(candidate)
    if not m:
        raise InvalidRepositoryURL(
            f"Not a GitHub repository URL: {candidate!r}. "
            "Expected the form https://github.com/<owner>/<repo>"
        )
    owner, name = m.group("owner"), m.group("repo")
    if name in {".", ".."} or name.startswith("-"):
        raise InvalidRepositoryURL(f"Invalid repository name: {name!r}")
    return RepoRef(url=f"https://github.com/{owner}/{name}", owner=owner, name=name)


def _git_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "GIT_TERMINAL_PROMPT": "0",  # never prompt -> private repos fail fast
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_ASKPASS": "true",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
        }
    )
    return env


def clone_repository(ref: RepoRef, timeout: int = 300, depth: int = 200) -> Path:
    """Shallow-clone ``ref`` into a new temporary directory and return its path."""
    if shutil.which("git") is None:
        raise CloneError("git executable not found on PATH; please install git.")
    workdir = Path(tempfile.mkdtemp(prefix="devops-audit-"))
    dest = workdir / ref.name
    cmd = [
        "git",
        "-c", "core.hooksPath=/dev/null",
        "-c", "protocol.file.allow=never",
        "-c", "core.symlinks=true",
        "clone",
        "--config", "core.autocrlf=false",
        "--depth", str(depth),
        "--no-tags",
        "--single-branch",
        "--no-recurse-submodules",
        "--",
        ref.url + ".git",
        str(dest),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=_git_env(), check=False
        )
    except subprocess.TimeoutExpired as exc:
        cleanup(workdir)
        raise CloneError(f"Cloning {ref.url} timed out after {timeout}s.") from exc
    if proc.returncode != 0:
        cleanup(workdir)
        msg = redact_text((proc.stderr or proc.stdout or "").strip())
        hint = ""
        if "could not read Username" in msg or "not found" in msg.lower() or "403" in msg:
            hint = " (the repository may not exist or may be private)"
        raise CloneError(f"git clone failed for {ref.url}{hint}: {msg[-500:]}")
    make_read_only(dest)
    return dest


def make_read_only(root: Path) -> None:
    """Remove write permission from the working tree (not .git, git needs it)."""
    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirnames:
            dirnames.remove(".git")
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                if not p.is_symlink():
                    mode = p.stat().st_mode
                    p.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            except OSError:
                pass


def working_tree_fingerprint(root: Path) -> str:
    """``git status --porcelain`` output; empty string means unchanged."""
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root, capture_output=True, text=True, timeout=60, env=_git_env(), check=False,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "<unavailable>"


def cleanup(path: Path) -> None:
    """Delete a temporary clone (restoring write permission first)."""
    if not path or not Path(path).exists():
        return

    def _onerror(func, p, _exc):  # noqa: ANN001
        try:
            os.chmod(p, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
            func(p)
        except OSError:
            pass

    for dirpath, dirnames, filenames in os.walk(path):
        for name in dirnames + filenames:
            try:
                p = Path(dirpath) / name
                if not p.is_symlink():
                    p.chmod(p.stat().st_mode | stat.S_IWUSR)
            except OSError:
                pass
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_onerror)
    else:  # pragma: no cover - older Pythons
        shutil.rmtree(path, onerror=_onerror)
