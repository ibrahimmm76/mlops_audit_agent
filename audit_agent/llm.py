"""Thin LLM provider adapters.

Only the official provider SDKs are used (``anthropic`` and ``openai``); there
is no agent framework.  The agent loop talks to a provider-neutral interface:

    LLMClient.complete(system, messages, tools) -> LLMResponse

Neutral message format used by the loop::

    {"role": "user", "content": str}
    {"role": "assistant", "content": str | None,
     "tool_calls": [{"id": str, "name": str, "arguments": dict}]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}

Supported providers (``LLM_PROVIDER``): anthropic, openai, gemini, groq,
openrouter, ollama.  Everything except Anthropic uses the OpenAI-compatible
chat-completions API.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .redact import redact_text


class LLMError(RuntimeError):
    """Fatal LLM/provider error (bad key, quota, unreachable after retries)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Opaque provider metadata that must be echoed back verbatim on the next request
    # (e.g. Gemini 3 thought signatures in ``extra_content``).
    provider_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse: ...


PROVIDERS: dict[str, dict[str, str | None]] = {
    "anthropic": {"key_env": "ANTHROPIC_API_KEY", "base_url": None, "model": "claude-sonnet-5-5"},
    "openai": {"key_env": "OPENAI_API_KEY", "base_url": None, "model": "gpt-4.1"},
    "gemini": {
        "key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.8-flash",
    },
    "groq": {
        "key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
    "openrouter": {
        "key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "anthropic/claude-sonnet-4.5",
    },
    "ollama": {"key_env": None, "base_url": "http://localhost:11434/v1", "model": "qwen2.5:14b"},
}

_TRANSIENT_HINTS = ("rate", "429", "500", "502", "503", "504", "overloaded", "timeout", "timed out",
                    "temporarily", "connection", "unavailable", "resource_exhausted")


def _is_overload_message(msg: str) -> bool:
    low = msg.lower()
    return any(h in low for h in ("503", "unavailable", "high demand", "overloaded", "timed out", "timeout",
                                  "429", "resource_exhausted", "rate",
                                  # model retired / not enabled for this key -> the fallback can still help
                                  "no longer available", "not_found", "notfounderror"))


def _is_timeout(exc: Exception) -> bool:
    return "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower()


def _retrying(fn, attempts: int = 6, base_delay: float = 2.0, max_timeouts: int = 2):  # noqa: ANN001,ANN202
    """Call ``fn`` with exponential backoff on transient provider errors.

    All retrying happens here (SDK-level retries are disabled) so the total wait
    stays bounded. Timeouts are retried at most ``max_timeouts`` times because
    each one already costs the full request timeout.
    """
    last: Exception | None = None
    timeouts = 0
    for i in range(attempts):
        try:
            return fn()
        except LLMError:
            raise
        except Exception as exc:  # provider SDK exceptions vary
            last = exc
            msg = str(exc).lower()
            status = getattr(exc, "status_code", None)
            if _is_timeout(exc):
                timeouts += 1
                if timeouts > max_timeouts:
                    break
                _notify(f"LLM request timed out; retrying ({timeouts}/{max_timeouts})")
                continue
            transient = (status in (408, 409, 429) or (isinstance(status, int) and status >= 500)
                         or any(h in msg for h in _TRANSIENT_HINTS))
            if status in (400, 401, 403, 404) or not transient or i == attempts - 1:
                break
            delay = min(base_delay * (2 ** i), 40.0)  # free tiers have per-minute limits
            _notify(f"LLM provider busy/rate-limited ({status or type(exc).__name__}); retrying in {delay:.0f}s")
            time.sleep(delay)
    raise LLMError(f"LLM request failed: {redact_text(type(last).__name__ + ': ' + str(last))[:500]}")


_GEMINI_SKIP_SIGNATURE = "skip_thought_signature_validator"


def _extra_content(tool_call: Any) -> dict[str, Any]:
    """Return ``{"extra_content": ...}`` if the provider attached it to a tool call."""
    extra = getattr(tool_call, "extra_content", None)
    if extra is None:
        model_extra = getattr(tool_call, "model_extra", None) or {}
        extra = model_extra.get("extra_content") if isinstance(model_extra, dict) else None
    if extra is not None and hasattr(extra, "model_dump"):
        extra = extra.model_dump()
    return {"extra_content": extra} if isinstance(extra, dict) and extra else {}


def _notify(msg: str) -> None:
    print(f"  ... {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
class AnthropicClient:
    provider = "anthropic"

    def __init__(self, api_key: str, model: str, max_tokens: int = 4096, timeout: float = 120.0,
                 temperature: float | None = None) -> None:
        try:
            import anthropic  # noqa: PLC0415 - optional dependency, imported lazily
        except ImportError as exc:
            raise LLMError("The 'anthropic' package is not installed: pip install -r requirements.txt") from exc
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=0)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @staticmethod
    def _convert(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            role = m["role"]
            if role == "user":
                block = {"type": "text", "text": m["content"]}
                if out and out[-1]["role"] == "user":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif role == "assistant":
                content: list[dict[str, Any]] = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls") or []:
                    content.append({"type": "tool_use", "id": tc["id"], "name": tc["name"],
                                    "input": tc["arguments"]})
                if not content:
                    content.append({"type": "text", "text": "(continuing)"})
                out.append({"role": "assistant", "content": content})
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def complete(self, system, messages, tools):  # noqa: ANN001,ANN201
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self._convert(messages),
            "tools": [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        resp = _retrying(lambda: self._client.messages.create(**kwargs))
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                args = block.input if isinstance(block.input, dict) else {}
                calls.append(ToolCall(id=block.id, name=block.name, arguments=args))
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {"input_tokens": getattr(resp.usage, "input_tokens", 0) or 0,
                     "output_tokens": getattr(resp.usage, "output_tokens", 0) or 0}
        return LLMResponse(text="\n".join(text_parts).strip(), tool_calls=calls,
                           stop_reason=getattr(resp, "stop_reason", None), usage=usage)


# --------------------------------------------------------------------------- #
# OpenAI-compatible (OpenAI, Gemini, Groq, OpenRouter, Ollama)
# --------------------------------------------------------------------------- #
class OpenAICompatibleClient:
    def __init__(self, provider: str, api_key: str | None, model: str, base_url: str | None,
                 max_tokens: int = 4096, timeout: float = 120.0, temperature: float | None = None,
                 reasoning_effort: str | None = None, fallback_model: str | None = None) -> None:
        try:
            import openai  # noqa: PLC0415 - optional dependency, imported lazily
        except ImportError as exc:
            raise LLMError("The 'openai' package is not installed: pip install -r requirements.txt") from exc
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Thinking models (e.g. Gemini 3.x) can spend minutes "thinking" per call; the agent makes
        # many small decisions, so low effort keeps each step fast. Dropped automatically if rejected.
        self.reasoning_effort = reasoning_effort or None
        # Used once if the primary model stays overloaded/unavailable, so a long audit is not lost.
        self.fallback_model = fallback_model if fallback_model and fallback_model != model else None
        self._client = openai.OpenAI(api_key=api_key or "not-needed", base_url=base_url,
                                     timeout=timeout, max_retries=0)

    @staticmethod
    def _convert(system: str, messages: list[dict[str, Any]], provider: str = "",
                 model: str | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = []
                    for idx, tc in enumerate(m["tool_calls"]):
                        call = {"id": tc["id"], "type": "function",
                                "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                        pdata = tc.get("provider_data") or {}
                        extra = pdata.get("extra_content")
                        # Signatures are only valid for the model that produced them (matters after a
                        # fallback switch); otherwise the placeholder below is used.
                        same_model = model is None or pdata.get("model") in (None, model)
                        if extra and same_model:
                            call["extra_content"] = extra  # echo Gemini thought signature verbatim
                        elif provider == "gemini" and idx == 0:
                            # Gemini 3 requires a signature on the first call of each turn; this documented
                            # placeholder is used only if the original signature is unavailable.
                            call["extra_content"] = {"google": {"thought_signature": _GEMINI_SKIP_SIGNATURE}}
                        msg["tool_calls"].append(call)
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        return out

    def complete(self, system, messages, tools):  # noqa: ANN001,ANN201
        try:
            return self._complete(system, messages, tools)
        except LLMError as exc:
            if not self.fallback_model or not _is_overload_message(str(exc)):
                raise
            _notify(f"model {self.model} unavailable; switching to fallback model {self.fallback_model}")
            self.model, self.fallback_model = self.fallback_model, None
            return self._complete(system, messages, tools)

    def _complete(self, system, messages, tools):  # noqa: ANN001,ANN202
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert(system, messages, self.provider, self.model),
            "tools": [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                        "parameters": t["parameters"]}} for t in tools],
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            resp = _retrying(lambda: self._client.chat.completions.create(**kwargs))
        except LLMError as exc:
            if "reasoning_effort" not in kwargs or "reasoning" not in str(exc).lower() or "400" not in str(exc):
                raise
            # This model does not support reasoning_effort: remember that and retry without it.
            _notify(f"model {self.model} rejected reasoning_effort; continuing without it")
            self.reasoning_effort = None
            kwargs.pop("reasoning_effort")
            resp = _retrying(lambda: self._client.chat.completions.create(**kwargs))
        if not resp.choices:
            raise LLMError("LLM returned no choices")
        choice = resp.choices[0]
        msg = choice.message
        calls: list[ToolCall] = []
        for i, tc in enumerate(getattr(msg, "tool_calls", None) or []):
            raw = tc.function.arguments or "{}"
            try:
                args = json.loads(raw)
                if not isinstance(args, dict):
                    args = {"__invalid_arguments__": raw[:500]}
            except json.JSONDecodeError:
                args = {"__invalid_arguments__": raw[:500]}
            pdata = _extra_content(tc)
            if pdata:
                pdata["model"] = self.model
            calls.append(ToolCall(id=tc.id or f"call_{i}", name=tc.function.name, arguments=args,
                                  provider_data=pdata))
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {"input_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                     "output_tokens": getattr(resp.usage, "completion_tokens", 0) or 0}
        return LLMResponse(text=(msg.content or "").strip(), tool_calls=calls,
                           stop_reason=choice.finish_reason, usage=usage)


def create_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    """Build a client from arguments and environment variables.

    Provider resolution: explicit argument > ``LLM_PROVIDER`` > first provider
    whose API key environment variable is set.
    """
    provider = (provider or os.getenv("LLM_PROVIDER") or "").strip().lower()
    if not provider:
        for name, cfg in PROVIDERS.items():
            if cfg["key_env"] and os.getenv(cfg["key_env"] or ""):
                provider = name
                break
    if not provider:
        raise LLMError(
            "No LLM configured. Set LLM_PROVIDER and the matching API key in .env "
            "(e.g. ANTHROPIC_API_KEY=...). See .env.example."
        )
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown LLM_PROVIDER {provider!r}; choose one of {sorted(PROVIDERS)}")

    cfg = PROVIDERS[provider]
    model = model or os.getenv("LLM_MODEL") or str(cfg["model"])
    key_env = cfg["key_env"]
    api_key = os.getenv(key_env) if key_env else None
    if key_env and not api_key:
        raise LLMError(f"{key_env} is not set (required for LLM_PROVIDER={provider}).")
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "8192"))
    timeout = float(os.getenv("LLM_TIMEOUT", "300"))
    temp_env = os.getenv("LLM_TEMPERATURE")
    temperature = float(temp_env) if temp_env not in (None, "") else None

    if provider == "anthropic":
        return AnthropicClient(api_key or "", model, max_tokens=max_tokens, timeout=timeout,
                               temperature=temperature)
    base_url = os.getenv("LLM_BASE_URL") or cfg["base_url"]
    effort = os.getenv("LLM_REASONING_EFFORT")
    if effort is None:
        effort = "low" if provider in ("gemini", "openai") else ""
    return OpenAICompatibleClient(provider, api_key, model, base_url, max_tokens=max_tokens,
                                  timeout=timeout, temperature=temperature,
                                  reasoning_effort=effort.strip().lower() or None,
                                  fallback_model=(os.getenv("LLM_FALLBACK_MODEL") or "").strip() or None)
