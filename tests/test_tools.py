import shutil
import unittest

from audit_agent.repo import cleanup, make_read_only, working_tree_fingerprint
from tests.helpers import FAKE_AWS_KEY, make_repo
from tools import RepoTools, ToolError

FAKE_ENV_TOKEN = "abcd1234" + "efgh5678"

WORKFLOW_BAD = """name: CI
on:
  push:
    branches: [master]
  pull_request_target:
permissions: write-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
      - uses: someone/action@main
      - run: echo "${{ github.event.pull_request.title }}"
      - run: pytest || true
      - run: echo ${{ secrets.DEPLOY_TOKEN }}
  deploy:
    runs-on: ubuntu-latest
    steps:
      - run: twine upload dist/*
"""

WORKFLOW_GOOD = """name: CI
on:
  push:
    branches: [main]
  pull_request:
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: pytest
"""


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bad_root = make_repo({
            "src/config.py": (
                "import os\n"
                f'API_KEY = "{FAKE_AWS_KEY}"\n'
                'DATA_DIR = "/home/alice/projects/data"\n'
                'PLACEHOLDER_PASSWORD = "changeme"\n'
                "def f():\n    try:\n        return 1\n    except:\n        pass\n"
            ),
            "requirements.txt": "requests\nflask>=2.0\nnumpy==1.21.0\n",
            "pyproject.toml": '[project]\nname = "x"\nversion = "0.1"\ndependencies = ["numpy>=1.24"]\n',
            "__pycache__/mod.cpython-311.pyc": b"\x00\x01binary",
            ".env": "SECRET_TOKEN=" + FAKE_ENV_TOKEN + "\n",
            ".env.example": "SECRET_TOKEN=your-token-here\n",
            ".github/workflows/ci.yml": WORKFLOW_BAD,
            "logo.png": b"\x89PNG\x00\x00",
        })
        cls.good_root = make_repo({
            "README.md": "# Demo\n",
            ".gitignore": "__pycache__/\n.env\n",
            "app.py": "def add(a, b):\n    return a + b\n",
            "tests/test_app.py": "from app import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
            "requirements.txt": "flask==3.0.0\n",
            "pyproject.toml": '[tool.ruff]\nline-length = 100\n',
            ".github/workflows/ci.yml": WORKFLOW_GOOD,
        })
        cls.bad = RepoTools(cls.bad_root)
        cls.good = RepoTools(cls.good_root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.bad_root, ignore_errors=True)
        shutil.rmtree(cls.good_root, ignore_errors=True)

    # --- required tools --------------------------------------------------
    def test_list_files_flags_generated_and_secret_files(self):
        out = self.bad.list_files()
        self.assertIn("src/config.py", out)
        self.assertIn("__pycache__/mod.cpython-311.pyc  <- python bytecode", out)
        self.assertIn(".env  <- local env/secret file", out)
        self.assertNotIn(".env.example  <-", out)  # templates are fine
        self.assertIn("__pycache__/mod.cpython-311.pyc", self.bad.observed)

    def test_list_files_pattern_and_path(self):
        self.assertIn("1 tracked file(s)", self.bad.list_files(pattern="*.py"))
        self.assertIn("ci.yml", self.bad.list_files(path=".github"))

    def test_read_file_numbers_lines_and_redacts(self):
        out = self.bad.read_file("src/config.py")
        self.assertIn("    3| DATA_DIR", out)
        self.assertNotIn(FAKE_AWS_KEY, out)
        self.assertIn("[REDACTED:aws_access_key_id]", out)
        self.assertIn(FAKE_AWS_KEY, self.bad.secret_values)  # remembered in memory only
        self.assertIn(3, self.bad.observed["src/config.py"])

    def test_read_file_ranges(self):
        out = self.bad.read_file("src/config.py", start_line=2, end_line=3)
        self.assertIn("lines 2-3 of", out)
        self.assertNotIn("    1|", out)

    def test_read_file_safety(self):
        for bad_path in ("../etc/passwd", "/etc/passwd", ".git/config", "missing.txt"):
            with self.assertRaises(ToolError, msg=bad_path):
                self.bad.read_file(bad_path)
        self.assertIn("binary file", self.bad.read_file("logo.png"))

    def test_search_code(self):
        out = self.bad.search_code(r"/home/|/Users/")
        self.assertIn("src/config.py:3:", out)
        self.assertIn("No matches", self.bad.search_code("zzz_not_present"))
        with self.assertRaises(ToolError):
            self.bad.search_code("([unclosed")

    def test_search_code_redacts(self):
        out = self.bad.search_code("API_KEY")
        self.assertNotIn(FAKE_AWS_KEY, out)

    def test_git_log(self):
        out = self.bad.git_log()
        self.assertIn("initial commit", out)
        self.assertIn("commits available locally", out)
        self.assertIn("src/config.py", self.bad.git_log(show_files=True))

    # --- optional tools --------------------------------------------------
    def test_git_status_and_show(self):
        self.assertIn("clean (unchanged)", self.bad.git_status())
        sha = self.bad.git_log().split()[0]
        show = self.bad.git_show_commit(sha)
        self.assertIn("initial commit", show)
        self.assertNotIn(FAKE_AWS_KEY, show)
        with self.assertRaises(ToolError):
            self.bad.git_show_commit("--output=/tmp/x")

    def test_inspect_workflows_flags_risks(self):
        out = self.bad.inspect_workflows()
        for expected in ("does not include the default branch 'main'", "write-all",
                         "checks out untrusted PR head", "mutable branch 'main'", "script injection",
                         "masks failures with '|| true'", "echoes a secret", "has no 'needs:' dependency"):
            self.assertIn(expected, out)

    def test_inspect_workflows_healthy(self):
        out = self.good.inspect_workflows()
        self.assertIn("FLAGS: none", out)
        self.assertIn("test-running jobs: ['test']", out)

    def test_inspect_workflows_invalid_yaml_falls_back_to_line_scan(self):
        broken = ("on: push\njobs:\n  t:\n    runs-on: ubuntu-latest\n    steps:\n"
                  "      - run: echo \"PR: ${{ github.event.pull_request.title }}\"\n"
                  "      - run: pytest || true\n")
        root = make_repo({".github/workflows/ci.yml": broken})
        try:
            out = RepoTools(root).inspect_workflows()
            self.assertIn("INVALID YAML (line 6)", out)
            self.assertIn("NONE of its jobs run", out)
            self.assertIn("line 7: masks failures", out)
            self.assertIn("line 6: untrusted github.event data", out)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_inspect_workflows_none(self):
        root = make_repo({"a.py": "x = 1\n", ".gitlab-ci.yml": "test:\n  script: pytest\n"})
        try:
            out = RepoTools(root).inspect_workflows()
            self.assertIn("No GitHub Actions workflows", out)
            self.assertIn(".gitlab-ci.yml", out)  # acceptable alternative reported
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_check_dependencies(self):
        out = self.bad.check_dependencies()
        self.assertIn("requests (line 1)", out)
        self.assertIn("numpy: requirements.txt -> ==1.21.0; pyproject.toml -> >=1.24", out)
        self.assertIn("Lockfiles present: none", out)

    def test_check_dependencies_package_lock_drift(self):
        root = make_repo({
            "package.json": '{"dependencies": {"express": "^4.0.0", "lodash": "*"}}',
            "package-lock.json": '{"packages": {"": {"dependencies": {"express": "^4.0.0"}}}}',
        })
        try:
            out = RepoTools(root).check_dependencies()
            self.assertIn("OUT OF SYNC", out)
            self.assertIn("lodash@*", out)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_detect_project_tooling(self):
        good = self.good.detect_project_tooling()
        self.assertIn("ruff (pyproject [tool.ruff])", good)
        self.assertIn("Test files: 1", good)
        bad = self.bad.detect_project_tooling()
        self.assertIn("Lint/format tooling: NONE detected", bad)
        self.assertIn(".gitignore: MISSING", bad)

    def test_scan_secrets(self):
        out = self.bad.scan_secrets()
        self.assertIn("src/config.py:2: aws_access_key_id", out)
        self.assertIn(".env:1:", out)
        self.assertNotIn(".env.example:1", out)  # placeholder ignored
        self.assertNotIn(FAKE_AWS_KEY, out)
        self.assertNotIn(FAKE_ENV_TOKEN, out)

    def test_run_ruff_or_unavailable(self):
        out = self.bad.run_ruff()
        if "not installed" not in out:
            self.assertIn("E722", out)

    def test_code_metrics(self):
        out = self.bad.code_metrics()
        self.assertIn("Bare 'except:' clauses: 1", out)
        root = make_repo({"broken.py": "def f(:\n  pass\n"})
        try:
            self.assertIn("broken.py:1", RepoTools(root).code_metrics())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_code_metrics_flags_missing_local_module(self):
        root = make_repo({"app/__init__.py": "", "app/core.py": "X = 1\n",
                          "tests/test_x.py": "from app.core import X\nfrom app.ghost import Y\nimport os\n"})
        try:
            out = RepoTools(root).code_metrics()
            self.assertIn("tests/test_x.py:2 imports 'app.ghost'", out)
            self.assertNotIn("app.core'", out)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_scan_secrets_finds_secret_in_deleted_file(self):
        secret = "Sup3r" + "Secret!Pass"
        root = make_repo({"creds.json": '{"db_password": "' + secret + '"}\n', "a.py": "x = 1\n"},
                         extra_commits=[{"delete": ["creds.json"], "message": "remove creds"}])
        try:
            out = RepoTools(root).scan_secrets()
            self.assertIn("SECRETS STILL IN GIT HISTORY", out)
            self.assertIn("creds.json:1", out)
            self.assertNotIn(secret, out)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_scan_secrets_separates_docs_and_tests_from_real_config(self):
        real = "Pr0d" + "Pa55w0rd!x"
        root = make_repo({
            "docs/guide.md": "Use a proxy: http://user:pass@host:8080 and SECRET_KEY = 'dev'\n",
            "tests/test_cfg.py": 'SECRET_KEY = "test key"\nTOKEN = "abc" + "defghijkl"\n',
            "package.json": '{"dependencies": {"pbkdf2-password": "1.2.1"}}\n',
            "app/settings.py": f'DB_PASSWORD = "{real}"\n',
        })
        try:
            out = RepoTools(root).scan_secrets()
            self.assertIn("1 likely real credential(s) in source/config files", out)
            self.assertIn("app/settings.py:1", out)
            self.assertNotIn("package.json", out)
            self.assertNotIn("docs/guide.md:1: connection", out.split("tests/docs/examples")[0])
            self.assertNotIn(real, out)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_scan_secrets_fast_on_huge_minified_line(self):
        import time
        root = make_repo({"static/app.min.js": "var a" + "_password_x" * 40000 + "=1;\n"})
        try:
            t0 = time.time()
            RepoTools(root).scan_secrets()
            self.assertLess(time.time() - t0, 5.0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_specs_have_schemas(self):
        names = {s.name for s in self.bad.specs()}
        self.assertTrue({"list_files", "read_file", "search_code", "git_log"} <= names)
        for s in self.bad.specs():
            self.assertEqual(s.parameters["type"], "object")

    def test_tools_never_modify_repository(self):
        root = make_repo({"a.py": "import os\n", ".github/workflows/x.yml": WORKFLOW_GOOD})
        try:
            make_read_only(root)
            before = working_tree_fingerprint(root)
            t = RepoTools(root)
            for spec in t.specs():
                if spec.name in ("read_file",):
                    spec.func("a.py")
                elif spec.name == "search_code":
                    spec.func("import")
                elif spec.name == "git_show_commit":
                    spec.func(t.git_log().split()[0])
                else:
                    spec.func()
            self.assertEqual(before, working_tree_fingerprint(root))
            self.assertEqual(before, "")
        finally:
            cleanup(root)


if __name__ == "__main__":
    unittest.main()
