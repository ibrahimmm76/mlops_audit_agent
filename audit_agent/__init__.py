"""DevOps repository auditing agent.

The package is split into small, testable modules:

* ``repo``     - GitHub URL validation and safe, read-only cloning
* ``redact``   - secret detection and redaction (nothing secret leaves the process)
* ``llm``      - thin provider adapters (Anthropic, OpenAI-compatible APIs)
* ``loop``     - the explicit Observe -> Reason -> Act agent loop
* ``report``   - finding validation, de-duplication and JSON report writing
* ``prompts``  - system prompt and rule catalogue
* ``trace``    - JSONL tool-call trace for auditing the agent's behaviour

The inspection tools themselves live in the top-level ``tools.py`` module.
"""

__version__ = "1.0.0"
