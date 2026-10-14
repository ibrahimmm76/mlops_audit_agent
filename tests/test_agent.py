"""Agent-loop and CLI behaviour tests using a scripted fake LLM (no network)."""

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import agent as cli
from audit_agent.llm import LLMError, LLMResponse
from audit_agent.loop import AuditAgent
from audit_agent.repo import CloneError
from audit_agent.trace import Tracer
from tests.helpers import FAKE_AWS_KEY, FULL_COVERAGE, ScriptedLLM, make_repo
from tools import RepoTools

SECRET_FINDING = {
    "rule_id": "SEC-001", "category": "security", "severity": "critical", "file": "config.py", "line": 2,
    "description": "Hardcoded AWS access key", "evidence": "AWS key literal assigned to API_KEY",
    "recommendation": "Load it from an environment variable and rotate the key",
}
README_FINDING = {
    "rule_id": "HYG-001", "category": "repository_hygiene", "severity": "medium", "file": "", "line": 0,
    "description": "No README", "evidence": "list_files shows no README file",
    "recommendation": "Add a README with setup and usage instructions",
}


def happy_script():
    return [
        [("list_files", {})],
        [("read_file", {"path": "config.py"}), ("scan_secrets", {})],
        [("record_violation", SECRET_FINDING), ("record_violation", README_FINDING)],
        [("finish", {"coverage": {"security": "scanned"}})],          # coverage gate rejects
        [("finish", {"coverage": FULL_COVERAGE})],                    # passes coverage, review gate
        [("finish", {"coverage": FULL_COVERAGE, "summary": "done"})],  # accepted
    ]


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.root = make_repo({"config.py": f'import os\nAPI_KEY = "{FAKE_AWS_KEY}"\n'})
        self.tools = RepoTools(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_llm_drives_tool_selection_and_gates(self):
        llm = ScriptedLLM(happy_script())
        tracer = Tracer()
        result = AuditAgent(llm, self.tools, tracer, max_steps=20, repo_url="https://github.com/a/b").run()

        self.assertTrue(result.finished)
        self.assertEqual(result.stop_reason, "finished")
        self.assertEqual(len(result.findings), 2)
        called = [e["name"] for e in tracer.events if e["event"] == "tool_call"]
        self.assertEqual(called[:3], ["list_files", "read_file", "scan_secrets"])
        gates = [e["message"] for e in tracer.events if e["event"] == "gate"]
        self.assertTrue(any("coverage gate" in g for g in gates))
        self.assertTrue(any("self-review gate" in g for g in gates))
        # every tool result was fed back to the LLM for interpretation
        last_msgs = llm.requests[-1]["messages"]
        self.assertTrue(any(m["role"] == "tool" and "config.py" in m["content"] for m in last_msgs))
        # the LLM never received the raw secret
        self.assertNotIn(FAKE_AWS_KEY, json.dumps(llm.requests))
        self.assertIn("record_violation", llm.requests[0]["tools"])

    def test_finding_requires_observed_evidence(self):
        llm = ScriptedLLM([
            [("record_violation", SECRET_FINDING)],                  # not inspected yet -> rejected
            [("read_file", {"path": "config.py"})],
            [("record_violation", SECRET_FINDING)],                  # accepted
            [("record_violation", {**SECRET_FINDING, "file": "nope.py"})],  # nonexistent file
            [("finish", {"coverage": FULL_COVERAGE})],
            [("finish", {"coverage": FULL_COVERAGE})],
        ])
        tracer = Tracer()
        result = AuditAgent(llm, self.tools, tracer, max_steps=10).run()
        self.assertEqual(len(result.findings), 1)
        rejected = [e for e in tracer.events if e["event"] == "finding_rejected"]
        self.assertEqual(len(rejected), 2)
        self.assertIn("not inspected", rejected[0]["error"])

    def test_withdraw_during_review(self):
        llm = ScriptedLLM([
            [("read_file", {"path": "config.py"}), ("record_violation", SECRET_FINDING)],
            [("finish", {"coverage": FULL_COVERAGE})],
            [("withdraw_violation", {"finding_id": "F1", "reason": "test fixture"})],
            [("finish", {"coverage": FULL_COVERAGE})],
        ])
        result = AuditAgent(llm, self.tools, Tracer(), max_steps=10).run()
        self.assertTrue(result.finished)
        self.assertEqual(result.findings, [])

    def test_unknown_tool_and_bad_arguments_are_recoverable(self):
        llm = ScriptedLLM([
            [("delete_everything", {}), ("read_file", {}), ("read_file", {"path": "../../etc/passwd"})],
            [("search_code", {"pattern": "(["})],
            [("finish", {"coverage": FULL_COVERAGE})],
            [("finish", {"coverage": FULL_COVERAGE})],
        ])
        tracer = Tracer()
        result = AuditAgent(llm, self.tools, tracer, max_steps=10).run()
        self.assertTrue(result.finished)
        outputs = [e["output"] for e in tracer.events if e["event"] == "tool_call"]
        self.assertIn("unknown tool", outputs[0])
        self.assertIn("missing required argument", outputs[1])
        self.assertIn("ERROR", outputs[2])
        self.assertIn("invalid regular expression", outputs[3])

    def test_max_steps_stops_gracefully(self):
        llm = ScriptedLLM([[("list_files", {})]] * 10)
        result = AuditAgent(llm, self.tools, Tracer(), max_steps=5).run()
        self.assertFalse(result.finished)
        self.assertEqual(result.stop_reason, "max_steps")
        self.assertEqual(result.steps, 5)

    def test_llm_that_stops_calling_tools(self):
        llm = ScriptedLLM([LLMResponse(text="I think we're done")] * 5)
        result = AuditAgent(llm, self.tools, Tracer(), max_steps=10).run()
        self.assertEqual(result.stop_reason, "llm_stopped")

    def test_llm_error_propagates(self):
        class Boom:
            provider, model = "fake", "boom"

            def complete(self, *a):
                raise LLMError("quota exceeded")

        with self.assertRaises(LLMError):
            AuditAgent(Boom(), self.tools, Tracer(), max_steps=5).run()


class CLITests(unittest.TestCase):
    def setUp(self):
        self.root = make_repo({"config.py": f'import os\nAPI_KEY = "{FAKE_AWS_KEY}"\n'})
        self.out_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def _run(self, argv, llm=None, cloner=None):
        err = io.StringIO()
        with redirect_stderr(err):
            code = cli.main(argv, llm_factory=(lambda p, m: llm) if llm else None,
                            cloner=cloner or (lambda ref, timeout: self.root))
        return code, err.getvalue()

    def test_end_to_end_writes_valid_report_and_trace(self):
        out = self.out_dir / "report.json"
        code, err = self._run(["--repo", "https://github.com/example/demo", "--output", str(out)],
                              llm=ScriptedLLM(happy_script()))
        self.assertEqual(code, 0, err)
        report = json.loads(out.read_text())
        self.assertEqual(report["repository"], "https://github.com/example/demo")
        self.assertEqual(len(report["violations"]), 2)
        self.assertEqual(report["violations"][0]["severity"], "critical")
        self.assertIsNone(report["violations"][1]["file"])
        self.assertIsNone(report["violations"][1]["line"])
        self.assertNotIn(FAKE_AWS_KEY, out.read_text())
        trace = (self.out_dir / "report.json.trace.jsonl").read_text()
        self.assertIn('"event": "tool_call"', trace)
        self.assertNotIn(FAKE_AWS_KEY, trace)

    def test_empty_violations_when_none_found(self):
        out = self.out_dir / "r.json"
        llm = ScriptedLLM([[("list_files", {})], [("finish", {"coverage": FULL_COVERAGE})],
                           [("finish", {"coverage": FULL_COVERAGE})]])
        code, _ = self._run(["--repo", "https://github.com/a/b", "--output", str(out), "--trace", "none"], llm=llm)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.read_text()), {"repository": "https://github.com/a/b", "violations": []})

    def test_invalid_url_exits_nonzero_without_report(self):
        out = self.out_dir / "r.json"
        code, err = self._run(["--repo", "https://gitlab.com/a/b", "--output", str(out)], llm=ScriptedLLM([]))
        self.assertEqual(code, cli.EXIT_INPUT)
        self.assertIn("Not a GitHub repository URL", err)
        self.assertFalse(out.exists())

    def test_clone_failure_exits_nonzero(self):
        def failing(ref, timeout):
            raise CloneError("repository not found")
        out = self.out_dir / "r.json"
        code, err = self._run(["--repo", "https://github.com/a/missing", "--output", str(out)],
                              llm=ScriptedLLM([]), cloner=failing)
        self.assertEqual(code, cli.EXIT_CLONE)
        self.assertIn("repository not found", err)
        self.assertFalse(out.exists())

    def test_llm_config_error_exits_nonzero(self):
        def no_llm(p, m):
            raise LLMError("GEMINI_API_KEY is not set")
        err = io.StringIO()
        with redirect_stderr(err):
            code = cli.main(["--repo", "https://github.com/a/b", "--output", str(self.out_dir / "r.json")],
                            llm_factory=no_llm, cloner=lambda r, t: self.root)
        self.assertEqual(code, cli.EXIT_LLM)
        self.assertIn("GEMINI_API_KEY", err.getvalue())

    def test_llm_runtime_error_exits_nonzero(self):
        class Boom:
            provider, model = "fake", "boom"

            def complete(self, *a):
                raise LLMError("503 unavailable")
        code, err = self._run(["--repo", "https://github.com/a/b", "--output", str(self.out_dir / "r.json")],
                              llm=Boom())
        self.assertEqual(code, cli.EXIT_LLM)
        self.assertFalse((self.out_dir / "r.json").exists())

    def test_missing_required_args(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            cli.main(["--repo", "https://github.com/a/b"])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
