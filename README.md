# DevOps Repository Auditing Agent

An LLM-driven agent that clones a public GitHub repository, **decides for itself** which inspection
tools to run, follows up on suspicious evidence, and writes an evidence-backed JSON report of DevOps
best-practice violations with actionable recommendations.

```
python agent.py --repo <github_repository_url> --output report.json
```

* Explicit, hand-written agent loop. **No** LangChain, LangGraph, n8n or other agent framework; only the
  official `anthropic` / `openai` SDKs are used to talk to the model.
* Works with **Gemini**, **Anthropic Claude**, **OpenAI**, **Groq**, **OpenRouter** or a local **Ollama**
  model. The provider is chosen in `.env`.
* Read-only and safe: the target is cloned into a temporary directory and made read-only. Repository code
  is never executed, and secret values never appear in logs, traces or the report.

---

## 1. Installation

Requirements: **Python 3.11+** and **git** on `PATH`.

```bash
git clone https://github.com/ibrahimmm76/mlops_audit_agent.git
cd mlops_audit_agent

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + pytest, for running the test suite
```

## 2. Setup (API credentials)

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Edit `.env` and fill in **one** provider:

| Provider | `.env` settings | Default model |
|---|---|---|
| Google Gemini | `LLM_PROVIDER=gemini`, `GEMINI_API_KEY=...` | `gemini-3.8-flash` |
| Anthropic | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY=...` | `claude-sonnet-5-5` |
| OpenAI | `LLM_PROVIDER=openai`, `OPENAI_API_KEY=...` | `gpt-4.1` |
| Groq | `LLM_PROVIDER=groq`, `GROQ_API_KEY=...` | `llama-3.3-70b-versatile` |
| OpenRouter | `LLM_PROVIDER=openrouter`, `OPENROUTER_API_KEY=...` | set `LLM_MODEL` |
| Ollama (local) | `LLM_PROVIDER=ollama` (no key) | `qwen2.5:14b` |

`LLM_MODEL=` overrides the default model. If `LLM_PROVIDER` is empty, the first provider whose key is
set is used. `.env` is git-ignored; never commit real keys.

## 3. Execution

```bash
python agent.py --repo https://github.com/<owner>/<repo> --output report.json
```

The run is fully unattended (no prompts). Progress goes to stderr; the report goes to `--output`.

| Option | Default | Meaning |
|---|---|---|
| `--repo` | *(required)* | Public GitHub repository URL |
| `--output` | *(required)* | Path of the JSON report |
| `--provider`, `--model` | from `.env` | Override the LLM provider/model |
| `--max-steps` | `60` | Maximum LLM reasoning steps |
| `--time-budget` | `1500` | Soft wall-clock limit (seconds) for the investigation |
| `--clone-timeout` | `300` | `git clone` timeout (seconds) |
| `--trace` | `<output>.trace.jsonl` | JSONL trace of every LLM decision and tool call (`none` disables) |
| `-v`, `--verbose` | off | Print the agent's reasoning and tool calls live |

**Exit codes**: `0` report written (the `violations` array may be empty) · `2` invalid input / not a GitHub
URL · `3` clone failed (missing/private repo, network, timeout) · `4` LLM configuration/provider error ·
`5` report could not be written · `1` unexpected error. A report file is only written for a completed
inspection.

### Output contract

```json
{
  "repository": "https://github.com/example/demo-project",
  "violations": [
    {
      "rule_id": "SEC-001",
      "category": "security",
      "severity": "critical",
      "file": "config.py",
      "line": 17,
      "description": "Hardcoded API credential",
      "evidence": "AWS access key literal assigned to API_KEY (value redacted)",
      "recommendation": "Load the key from an environment variable or secrets manager and rotate it"
    }
  ]
}
```

* `severity` ∈ `critical | high | medium | low`; `file`/`line` are JSON `null` when not applicable.
* `category` ∈ `repository_hygiene, dependency_management, code_quality, testing, ci, security,
  configuration, collaboration, build_release`.
* The report is validated against this schema before being written atomically.
* `sample_report.json` in this repository is an unedited output of the agent, produced with
  `python agent.py --repo <url> --output sample_report.json`. Its trace is kept next to it as
  `sample_report.json.trace.jsonl`.

---

## 4. Architecture

```
agent.py                CLI: argument parsing, exit codes, orchestration, cleanup
tools.py                Inspection tools (read-only) + JSON-schema tool registry
audit_agent/
  loop.py               Explicit agent loop, tool dispatch, finding gates
  llm.py                Provider adapters (Anthropic SDK, OpenAI-compatible SDK), retries
  prompts.py            System prompt, rule catalogue, gate messages, control-tool schemas
  report.py             Finding validation, de-duplication, report schema + atomic writing
  redact.py             Secret detection / redaction used on every output path
  repo.py               GitHub URL validation, safe shallow clone, read-only working tree
  trace.py              JSONL trace of decisions and tool calls
tests/                  Offline unit tests (scripted fake LLM, throwaway git repos)
```

### The agent loop (Observe → Reason → Select Tool → Execute → Interpret → Report)

```
            ┌───────────────────────────────────────────────────────────┐
            │ conversation = [task]                                     │
            ▼                                                           │
  LLM(system prompt, conversation, tool schemas)                        │
            │  returns reasoning text + chosen tool call(s)             │
            ▼                                                           │
  controller validates args ──► execute tool (read-only) ──► redact,    │
            │                                               truncate    │
            ▼                                                           │
  observation appended to conversation ─────────────────────────────────┘
            │
   record_violation ─► validate (enum values, file tracked, line in range,
            │          file actually inspected, no duplicate, no secrets)
            ▼
   finish ─► coverage gate (every category must be accounted for)
         ─► self-review gate (LLM re-checks each finding, withdraws weak ones)
         ─► report.json
```

1. **Observe.** The task message tells the LLM the repo URL, the default branch and the file count. Nothing
   else is pre-computed: the model must discover the repository through tools.
2. **Reason and select.** On every step the LLM receives all tool schemas and chooses the next call(s),
   with a one-sentence justification. There is no fixed checklist; the model decides the order, the
   arguments (which file to read, which regex to search) and the follow-ups.
3. **Execute.** `AuditAgent._dispatch` validates the arguments against the schema, runs the tool,
   catches errors (returned to the model as `ERROR: ...` so it can recover), redacts secrets and
   truncates long output.
4. **Interpret.** The observation goes back into the conversation. The model reads it and decides whether to
   dig deeper (for example, `inspect_workflows` flags a step, then the model calls `read_file` on that
   workflow), record a finding, or move on.
5. **Report.** Findings are recorded through the `record_violation` control tool. They are accepted only
   if they pass validation; the rejection message tells the model what to fix. `finish` must pass two
   gates before the report is written.

**Precision safeguards** (the hidden evaluation penalises false positives):
* A finding about a file is rejected unless the agent has inspected that file through a tool.
* The file must be tracked in the repo, and the line must exist.
* Duplicates (same rule/file/line, or the same problem in the same file) are rejected.
* The system prompt encodes the assignment's evidence rule: acceptable alternatives (flake8 vs ruff,
  GitLab CI vs Actions, poetry.lock vs pinned requirements...), placeholder vs real secrets, libraries vs
  applications for lockfiles, and "never infer branch protection from files".
* Tool heuristics are presented as *flags to confirm*, not verdicts.
* The self-review gate makes the model re-justify every finding and withdraw unsupported ones.

**Recall safeguards:** a rule catalogue covering all nine inspection areas, purpose-built tools for the
subtle defect classes (workflow security, dependency drift, secrets in history), and a coverage gate that
refuses `finish` until every category has been examined or explicitly ruled out.

**Reliability:** step and time budgets with warnings to the model, history compaction for long audits,
exponential-backoff retries for rate limits and 5xx errors, fast failure for authentication errors, and a
clean non-zero exit with a useful message on fatal errors.

### Tools

| Tool | Required? | What it does |
|---|---|---|
| `list_files(path, pattern, max_entries)` | ✔ | `git ls-files` listing with sizes; flags tracked generated/local/secret files (`__pycache__`, `.env`, venv, IDE settings, logs, build output) and large binaries |
| `read_file(path, start_line, end_line)` | ✔ | Line-numbered file content (400 lines per call), secrets redacted, path-traversal and symlink-escape safe |
| `search_code(pattern, path_glob, case_insensitive, max_results)` | ✔ | Regex search over tracked text files (`file:line: text`) |
| `git_log(max_count, path, show_files)` | ✔ | Commit history, optionally with changed files per commit |
| `git_status()` | | Default branch, HEAD, proof that the clone is unmodified |
| `git_show_commit(commit, path)` | | One commit's diff (redacted), e.g. a secret added then deleted |
| `inspect_workflows()` | | Parses GitHub Actions YAML with line numbers: triggers vs default branch, permissions, `pull_request_target` + PR checkout, script injection, `continue-on-error` / `\|\| true`, secrets echoed, mutable action refs, deploy jobs without a test gate, auto-merge |
| `check_dependencies()` | | requirements/pyproject/setup.py/Pipfile/package.json/go.mod/Cargo: constraint strength, conflicting specs across files, missing or out-of-sync lockfiles, insecure indexes |
| `detect_project_tooling()` | | Linters/formatters, test files and config, CI systems, packaging, docs and collaboration files, used to recognise **acceptable alternatives** |
| `scan_secrets(path_glob)` | | Credential detection by type and location; values are never revealed; placeholders and test fixtures are distinguished |
| `run_ruff(path)` | | Static ruff lint with a curated rule set (syntax errors, undefined names, bare except, hardcoded passwords, `verify=False`...) |
| `code_metrics()` | | AST metrics: syntax errors, giant functions/files, bare `except:`, leftover `breakpoint()` |

Control tools exposed to the LLM: `record_violation`, `withdraw_violation`, `finish`.

### Safety

* Only `https://github.com/<owner>/<repo>` URLs are accepted (validated; no local paths or option injection).
* Clone with `GIT_TERMINAL_PROMPT=0`, hooks disabled, LFS smudge and submodules off, then `chmod -w` on the
  working tree. `git status` is fingerprinted before and after the audit. Nothing is ever pushed, so the
  inspected repository is left unchanged. The temporary clone is deleted at the end.
* **No untrusted code execution.** Python is analysed with `ast.parse`, YAML with `yaml.safe_load`, and
  ruff only parses source. Running a target's test suite would mean executing arbitrary code (plus
  installing its dependencies), so `run_pytest` is intentionally **not** provided; test health is judged
  statically (tests exist, CI runs them, failures are not masked).
* All tool output, LLM text, traces and report fields pass through the redactor. Known secret values seen
  by the scanner are additionally scrubbed from findings.

### Rule catalogue

`HYG-001..005` hygiene · `DEP-001..006` dependencies · `CQ-001..003` code quality · `TST-001..004` testing ·
`CI-001..005` GitHub Actions/CI · `SEC-001..007` security · `CFG-001..004` configuration ·
`COL-001..002` collaboration · `BLD-001..004` build & release. Full descriptions are in
[`audit_agent/prompts.py`](audit_agent/prompts.py).

---

## 5. Tests

```bash
pytest                    # or: python -m unittest discover -s tests -t .
ruff check .
```

The tests run offline. A scripted fake LLM drives the real agent loop, and throwaway git repositories
with injected defects exercise every tool. They cover:

* each tool on a defective repo and a healthy repo (false-positive checks), including path traversal,
  binary files, bad regexes and read-only guarantees;
* the agent loop: LLM-chosen tool order, unknown tools and bad arguments, evidence-gated findings,
  coverage and self-review gates, withdrawals, step budget, LLM failures;
* provider adapters against stubbed SDKs (message conversion, tool-call parsing, retries, no key leakage);
* report schema, `null` handling, de-duplication, atomic writing, secret scrubbing;
* the CLI end to end: valid report and trace, empty-violations report, non-zero exits for invalid URL,
  clone failure and LLM errors.

CI (`.github/workflows/ci.yml`) runs ruff, pytest and a CLI smoke test on Python 3.11 to 3.13.

## 6. Self-evaluation (precision / recall)

[`ibrahimmm76/devops_tests`](https://github.com/ibrahimmm76/devops_tests) is a small Flask service with
26 deliberately injected defects across all nine categories. They range from obvious (hardcoded
credentials, tracked `.env`) to subtle (a secret deleted but still in git history, a CI branch filter that
never matches `main`, `pull_request_target` running PR code). It also has healthy parts that must *not*
be reported. The answer key is in [`evaluation/devops_tests_answer_key.json`](evaluation/devops_tests_answer_key.json).

```bash
python agent.py --repo https://github.com/ibrahimmm76/devops_tests --output sample_report.json -v
python evaluation/score.py sample_report.json evaluation/devops_tests_answer_key.json
```

The scorer uses the assignment's definitions. Precision = matched / reported and recall = matched /
expected. A match needs an accepted rule id, the same file, and a line within ±3; duplicates earn nothing.
It lists matched items, unmatched reports and missed defects, so you can see exactly where the agent
gained or lost points.

## 7. Inspecting the agent's behaviour

Every run writes `<output>.trace.jsonl`. Each line is one event: the LLM's reasoning text, each tool call
with arguments, duration and (redacted) output, findings recorded, rejected or withdrawn, and gate
decisions. Use `-v` to watch it live:

```bash
python agent.py --repo https://github.com/<owner>/<repo> --output report.json -v
```

## 8. Limitations

* Static analysis only: failing tests are detected when visibly broken or masked, not by running them.
* Shallow clone (latest 200 commits): very old history is not searched.
* Remote settings (branch protection, repo-level Actions permissions, secret scanning) cannot be seen from
  files and are deliberately not reported.
* Findings depend on the chosen model. Stronger models give better precision and recall; the agent's
  gates and validation keep weaker models from inventing evidence.
