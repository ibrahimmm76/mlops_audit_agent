"""Provider adapter tests with stubbed SDK modules (no network, no API keys)."""

import os
import sys
import types
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from audit_agent import llm as llm_mod
from audit_agent.llm import LLMError, create_client

CONVERSATION = [
    {"role": "user", "content": "Audit repo"},
    {"role": "assistant", "content": "Listing files first.",
     "tool_calls": [{"id": "c1", "name": "list_files", "arguments": {}},
                    {"id": "c2", "name": "read_file", "arguments": {"path": "a.py"}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "list_files", "content": "a.py"},
    {"role": "tool", "tool_call_id": "c2", "name": "read_file", "content": "1| x = 1"},
]
TOOLS = [{"name": "list_files", "description": "d", "parameters": {"type": "object", "properties": {}}}]


def _fake_anthropic_module(captured):
    class Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return NS(content=[NS(type="text", text="I will read it."),
                               NS(type="tool_use", id="tu1", name="read_file", input={"path": "b.py"})],
                      stop_reason="tool_use", usage=NS(input_tokens=10, output_tokens=5))

    class Anthropic:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.messages = Messages()

    return types.SimpleNamespace(Anthropic=Anthropic)


def _fake_openai_module(captured, arguments='{"path": "b.py"}'):
    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            call = NS(id="tc1", function=NS(name="read_file", arguments=arguments))
            return NS(choices=[NS(message=NS(content="Reading b.py", tool_calls=[call]), finish_reason="tool_calls")],
                      usage=NS(prompt_tokens=7, completion_tokens=3))

    class OpenAI:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.chat = NS(completions=Completions())

    return types.SimpleNamespace(OpenAI=OpenAI)


class AdapterTests(unittest.TestCase):
    def test_anthropic_message_conversion_and_parsing(self):
        captured = {}
        with mock.patch.dict(sys.modules, {"anthropic": _fake_anthropic_module(captured)}):
            client = llm_mod.AnthropicClient("key", "claude-test")
            resp = client.complete("SYS", CONVERSATION, TOOLS)
        msgs = captured["messages"]
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user"])
        self.assertEqual(msgs[1]["content"][1]["type"], "tool_use")
        # both tool results are grouped into ONE user message, as the API requires
        self.assertEqual([b["tool_use_id"] for b in msgs[2]["content"]], ["c1", "c2"])
        self.assertEqual(captured["tools"][0]["input_schema"], TOOLS[0]["parameters"])
        self.assertEqual(captured["system"], "SYS")
        self.assertEqual(resp.tool_calls[0].name, "read_file")
        self.assertEqual(resp.tool_calls[0].arguments, {"path": "b.py"})
        self.assertEqual(resp.text, "I will read it.")

    def test_openai_compatible_conversion_and_parsing(self):
        captured = {}
        with mock.patch.dict(sys.modules, {"openai": _fake_openai_module(captured)}):
            client = llm_mod.OpenAICompatibleClient("gemini", "k", "gemini-x", "https://example/v1")
            resp = client.complete("SYS", CONVERSATION, TOOLS)
        msgs = captured["messages"]
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "tool", "tool"])
        self.assertEqual(msgs[2]["tool_calls"][1]["function"]["arguments"], '{"path": "a.py"}')
        self.assertEqual(captured["tools"][0]["type"], "function")
        self.assertEqual(captured["init"]["base_url"], "https://example/v1")
        self.assertEqual(resp.tool_calls[0].arguments, {"path": "b.py"})
        self.assertEqual(resp.usage, {"input_tokens": 7, "output_tokens": 3})

    def test_openai_invalid_json_arguments_flagged(self):
        captured = {}
        with mock.patch.dict(sys.modules, {"openai": _fake_openai_module(captured, arguments="{not json")}):
            client = llm_mod.OpenAICompatibleClient("groq", "k", "m", None)
            resp = client.complete("SYS", CONVERSATION, TOOLS)
        self.assertIn("__invalid_arguments__", resp.tool_calls[0].arguments)

    def test_retry_gives_up_on_auth_error_without_leaking_key(self):
        class AuthError(Exception):
            status_code = 401

        def boom():
            raise AuthError("invalid api key sk-ant-abcdefghijklmnopqrstuvwxyz123")

        with self.assertRaises(LLMError) as ctx:
            llm_mod._retrying(boom, attempts=3, base_delay=0)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz123", str(ctx.exception))

    def test_retry_recovers_from_transient_error(self):
        calls = {"n": 0}

        class RateLimit(Exception):
            status_code = 429

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RateLimit("rate limited")
            return "ok"

        self.assertEqual(llm_mod._retrying(flaky, attempts=4, base_delay=0), "ok")


class TimeoutAndEffortTests(unittest.TestCase):
    def test_timeouts_retried_but_bounded(self):
        calls = {"n": 0}

        class APITimeoutError(Exception):
            pass

        def slow():
            calls["n"] += 1
            raise APITimeoutError("Request timed out.")

        with mock.patch("sys.stderr"), self.assertRaises(LLMError):
            llm_mod._retrying(slow, attempts=6, base_delay=0, max_timeouts=2)
        self.assertEqual(calls["n"], 3)  # first try + 2 retries, then give up

    def test_timeout_then_success(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("timed out")
            return "ok"

        with mock.patch("sys.stderr"):
            self.assertEqual(llm_mod._retrying(flaky, base_delay=0), "ok")

    def test_reasoning_effort_sent_and_dropped_if_rejected(self):
        sent = []

        class BadRequest(Exception):
            status_code = 400

        class Completions:
            def create(self, **kwargs):
                sent.append(dict(kwargs))
                if "reasoning_effort" in kwargs:
                    raise BadRequest("Error code: 400 - unknown field reasoning_effort")
                call = NS(id="t1", function=NS(name="list_files", arguments="{}"))
                return NS(choices=[NS(message=NS(content="", tool_calls=[call]), finish_reason="tool_calls")],
                          usage=None)

        fake = types.SimpleNamespace(OpenAI=lambda **kw: NS(chat=NS(completions=Completions())))
        with mock.patch.dict(sys.modules, {"openai": fake}), mock.patch("sys.stderr"):
            client = llm_mod.OpenAICompatibleClient("gemini", "k", "m", None, reasoning_effort="low")
            resp = client.complete("SYS", CONVERSATION, TOOLS)
            client.complete("SYS", CONVERSATION, TOOLS)
        self.assertEqual(sent[0]["reasoning_effort"], "low")
        self.assertNotIn("reasoning_effort", sent[1])
        self.assertNotIn("reasoning_effort", sent[2])  # remembered for later calls
        self.assertEqual(resp.tool_calls[0].name, "list_files")


class ThoughtSignatureTests(unittest.TestCase):
    """Gemini 3 rejects follow-up requests unless tool-call thought signatures are echoed back."""

    def test_signature_captured_and_echoed(self):
        sent = []
        sig = {"google": {"thought_signature": "SIG-abc123"}}

        class Completions:
            def create(self, **kwargs):
                sent.append(kwargs)
                call = NS(id="t1", function=NS(name="list_files", arguments="{}"), extra_content=sig)
                return NS(choices=[NS(message=NS(content="", tool_calls=[call]), finish_reason="tool_calls")],
                          usage=None)

        fake = types.SimpleNamespace(OpenAI=lambda **kw: NS(chat=NS(completions=Completions())))
        with mock.patch.dict(sys.modules, {"openai": fake}):
            client = llm_mod.OpenAICompatibleClient("gemini", "k", "m", None)
            resp = client.complete("SYS", [{"role": "user", "content": "go"}], TOOLS)
            tc = resp.tool_calls[0]
            self.assertEqual(tc.provider_data, {"extra_content": sig, "model": "m"})
            history = [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments,
                                 "provider_data": tc.provider_data}]},
                {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": "a.py"},
            ]
            client.complete("SYS", history, TOOLS)
        echoed = sent[1]["messages"][2]["tool_calls"][0]
        self.assertEqual(echoed["extra_content"], sig)

    def test_placeholder_only_for_gemini_first_call_without_signature(self):
        msgs = [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "name": "x", "arguments": {}}, {"id": "b", "name": "y", "arguments": {}}]}]
        gem = llm_mod.OpenAICompatibleClient._convert("S", msgs, "gemini")[1]["tool_calls"]
        self.assertEqual(gem[0]["extra_content"]["google"]["thought_signature"],
                         "skip_thought_signature_validator")
        self.assertNotIn("extra_content", gem[1])
        groq = llm_mod.OpenAICompatibleClient._convert("S", msgs, "groq")[1]["tool_calls"]
        self.assertNotIn("extra_content", groq[0])


class FallbackModelTests(unittest.TestCase):
    def test_switches_to_fallback_on_overload_and_drops_stale_signatures(self):
        sent = []

        class Unavailable(Exception):
            status_code = 503

        class Completions:
            def create(self, **kwargs):
                sent.append(kwargs)
                if kwargs["model"] == "busy-model":
                    raise Unavailable("Error code: 503 - This model is currently experiencing high demand")
                call = NS(id="t2", function=NS(name="read_file", arguments='{"path": "a.py"}'))
                return NS(choices=[NS(message=NS(content="", tool_calls=[call]), finish_reason="tool_calls")],
                          usage=None)

        history = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "t1", "name": "list_files", "arguments": {},
                "provider_data": {"extra_content": {"google": {"thought_signature": "OLD"}},
                                  "model": "busy-model"}}]},
            {"role": "tool", "tool_call_id": "t1", "name": "list_files", "content": "a.py"},
        ]
        fake = types.SimpleNamespace(OpenAI=lambda **kw: NS(chat=NS(completions=Completions())))
        with mock.patch.dict(sys.modules, {"openai": fake}), mock.patch("sys.stderr"), \
                mock.patch("audit_agent.llm.time.sleep"):
            client = llm_mod.OpenAICompatibleClient("gemini", "k", "busy-model", None,
                                                    fallback_model="spare-model")
            resp = client.complete("SYS", history, TOOLS)
        self.assertEqual(client.model, "spare-model")
        self.assertEqual(resp.tool_calls[0].name, "read_file")
        last = sent[-1]
        self.assertEqual(last["model"], "spare-model")
        sig = last["messages"][2]["tool_calls"][0]["extra_content"]["google"]["thought_signature"]
        self.assertEqual(sig, "skip_thought_signature_validator")  # old model's signature not reused

    def test_no_fallback_for_auth_errors(self):
        class AuthError(Exception):
            status_code = 401

        class Completions:
            def create(self, **kwargs):
                raise AuthError("Error code: 401 - API key not valid")

        fake = types.SimpleNamespace(OpenAI=lambda **kw: NS(chat=NS(completions=Completions())))
        with mock.patch.dict(sys.modules, {"openai": fake}), mock.patch("sys.stderr"):
            client = llm_mod.OpenAICompatibleClient("gemini", "k", "m1", None, fallback_model="m2")
            with self.assertRaises(LLMError):
                client.complete("SYS", [{"role": "user", "content": "go"}], TOOLS)
        self.assertEqual(client.model, "m1")


class FactoryTests(unittest.TestCase):
    def test_missing_configuration(self):
        env = {k: "" for k in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
                               "GROQ_API_KEY", "OPENROUTER_API_KEY")}
        with mock.patch.dict(os.environ, env):
            with self.assertRaises(LLMError):
                create_client()
            with self.assertRaises(LLMError):
                create_client("gemini")          # key missing
            with self.assertRaises(LLMError):
                create_client("not-a-provider")

    def test_gemini_selected_from_env(self):
        captured = {}
        env = {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key", "LLM_MODEL": ""}
        with mock.patch.dict(os.environ, env), \
                mock.patch.dict(sys.modules, {"openai": _fake_openai_module(captured)}):
            client = create_client()
        self.assertEqual(client.provider, "gemini")
        self.assertIn("generativelanguage.googleapis.com", captured["init"]["base_url"])


if __name__ == "__main__":
    unittest.main()
