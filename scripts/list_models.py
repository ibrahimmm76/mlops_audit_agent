#!/usr/bin/env python3
"""List the models your configured API key can use.

    python scripts/list_models.py            # provider from .env (LLM_PROVIDER)
    python scripts/list_models.py gemini

Pick a model id from the output and set it in .env as LLM_MODEL (and optionally a second
one as LLM_FALLBACK_MODEL). The key itself is never printed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_agent.llm import PROVIDERS  # noqa: E402


def main(argv: list[str]) -> int:
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    provider = (argv[1] if len(argv) > 1 else os.getenv("LLM_PROVIDER", "gemini")).lower()
    if provider not in PROVIDERS:
        print(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
        return 2
    cfg = PROVIDERS[provider]
    key = os.getenv(cfg["key_env"]) if cfg["key_env"] else "not-needed"
    if not key:
        print(f"{cfg['key_env']} is not set in .env")
        return 2
    if provider == "anthropic":
        import anthropic  # noqa: PLC0415
        models = [m.id for m in anthropic.Anthropic(api_key=key).models.list(limit=100)]
    else:
        import openai  # noqa: PLC0415
        client = openai.OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL") or cfg["base_url"], timeout=60)
        models = [m.id for m in client.models.list()]
    names = sorted({m.split("/")[-1] for m in models})
    if provider == "gemini":
        names = [n for n in names if n.startswith("gemini")]
    print(f"{len(names)} model(s) available for {provider}:")
    for n in names:
        print("  " + n)
    if provider == "gemini":
        print("\nTip: 'flash' / 'flash-lite' models are fastest. Example .env lines:\n"
              "  LLM_MODEL=<a flash model above>\n  LLM_FALLBACK_MODEL=<another model above>")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
