from __future__ import annotations

import copy
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import answer_question  # noqa: E402
import ask_agent  # noqa: E402


def tool_response(name: str = "retrieve_pdf", arguments: str = '{"query":"PDF fact?"}') -> dict[str, Any]:
    return {
        "choices": [{"message": {
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }],
        }}],
    }


def invalid_tool_responses() -> list[tuple[str, Any]]:
    cases: list[tuple[str, Any]] = [
        ("non-object response", None),
        ("missing choices", {}),
        ("non-list choices", {"choices": {"0": {}}}),
        ("empty choices", {"choices": []}),
        ("non-object choice", {"choices": [None]}),
        ("missing message", {"choices": [{}]}),
        ("non-object message", {"choices": [{"message": None}]}),
        ("multiple choices", {"choices": [{"message": {}}, {"message": {}}]}),
    ]
    for content in ({}, [], 123):
        response = tool_response()
        response["choices"][0]["message"]["content"] = content
        cases.append((f"invalid message content: {content!r}", response))
    response = tool_response("document_status", "{}")
    del response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    cases.append(("missing status arguments", response))
    for label, calls in [
        ("missing tool calls", None),
        ("non-list tool calls", {}),
        ("zero tool calls", []),
        ("multiple tool calls", tool_response()["choices"][0]["message"]["tool_calls"] * 2),
        ("non-object tool call", [None]),
    ]:
        response = tool_response()
        message = response["choices"][0]["message"]
        if label == "missing tool calls":
            del message["tool_calls"]
        else:
            message["tool_calls"] = calls
        cases.append((label, response))
    for field, values in [
        ("id", [None, "", "   ", 123]),
        ("type", [None, "tool", 123]),
        ("function", [None, [], "retrieve_pdf"]),
    ]:
        for value in values:
            response = tool_response()
            response["choices"][0]["message"]["tool_calls"][0][field] = value
            cases.append((f"invalid {field}: {value!r}", response))
        response = tool_response()
        del response["choices"][0]["message"]["tool_calls"][0][field]
        cases.append((f"missing {field}", response))
    for field, values in [
        ("name", [None, "", "run_shell", [], 123]),
        ("arguments", [None, {}, [], 123]),
    ]:
        for value in values:
            response = tool_response()
            response["choices"][0]["message"]["tool_calls"][0]["function"][field] = value
            cases.append((f"invalid function {field}: {value!r}", response))
        response = tool_response()
        del response["choices"][0]["message"]["tool_calls"][0]["function"][field]
        cases.append((f"missing function {field}", response))
    for label, name, arguments in [
        ("malformed JSON", "retrieve_pdf", "{"),
        ("trailing JSON", "retrieve_pdf", '{"query":"x"} {}'),
        ("JSON array", "retrieve_pdf", "[]"),
        ("JSON null", "retrieve_pdf", "null"),
        ("JSON string", "retrieve_pdf", '"query"'),
        ("duplicate JSON keys", "retrieve_pdf", '{"query":"x","query":"y"}'),
        ("NaN JSON constant", "retrieve_pdf", '{"query":NaN}'),
        ("Infinity JSON constant", "retrieve_pdf", '{"query":Infinity}'),
        ("negative Infinity JSON constant", "retrieve_pdf", '{"query":-Infinity}'),
        ("missing query", "retrieve_pdf", "{}"),
        ("non-string query", "retrieve_pdf", '{"query":123}'),
        ("extra query argument", "retrieve_pdf", '{"query":"x","path":"outside.pdf"}'),
        ("empty query", "retrieve_pdf", '{"query":""}'),
        ("whitespace query", "retrieve_pdf", '{"query":"   "}'),
        ("overlong query", "retrieve_pdf", json.dumps({"query": "x" * 2001})),
        ("status extra argument", "document_status", '{"query":"x"}'),
        ("status duplicate JSON keys", "document_status", '{"x":1,"x":2}'),
    ]:
        cases.append((label, tool_response(name, arguments)))
    return cases


TOOL_SETTINGS = {
    "chunks_json_path": Path("unused-chunks.json"),
    "top_k": 2,
    "candidate_k": 20,
    "embedding_model": "mock-embedding",
    "rerank_model": "mock-rerank",
    "max_context_chars": 3000,
    "max_chunk_chars": 1200,
    "local_files_only": True,
}
RUN_SETTINGS = {
    **TOOL_SETTINGS,
    "query": "PDF fact?",
    "api_key": "test-key",
    "base_url": "https://example.test",
    "llm_model": "test-model",
    "max_tokens": 600,
    "temperature": 0.2,
    "thinking": "disabled",
    "reasoning_effort": "high",
    "timeout_seconds": 60,
}


class AgentToolValidationTests(unittest.TestCase):
    def test_parse_both_approved_tools_and_normalize_query(self) -> None:
        for name, arguments, expected in [
            ("retrieve_pdf", '{"query":"  PDF fact?  "}', {"query": "PDF fact?"}),
            ("document_status", "{}", {}),
        ]:
            with self.subTest(tool=name):
                parsed = ask_agent.parse_tool_call(tool_response(name, arguments))
                self.assertEqual(parsed["name"], name)
                self.assertEqual(parsed["arguments"], expected)
                self.assertEqual(parsed["tool_call_id"], "call_1")
                self.assertEqual(parsed["assistant_message"]["role"], "assistant")

    def test_invalid_calls_are_rejected_before_any_tool_execution(self) -> None:
        for label, response in invalid_tool_responses():
            with self.subTest(case=label), \
                 patch.object(ask_agent, "call_deepseek", return_value=response) as model, \
                 patch.object(ask_agent, "execute_tool") as execute:
                with self.assertRaises(RuntimeError):
                    ask_agent.run_agent(**RUN_SETTINGS)
                model.assert_called_once()
                execute.assert_not_called()

    def test_executor_independently_rejects_unapproved_names_and_invalid_arguments(self) -> None:
        cases = [
            ("run_shell", {"query": "x"}),
            (None, {"query": "x"}),
            ([], {"query": "x"}),
            ("retrieve_pdf", None),
            ("retrieve_pdf", []),
            ("retrieve_pdf", {}),
            ("retrieve_pdf", {"query": 123}),
            ("retrieve_pdf", {"query": "x", "extra": True}),
            ("retrieve_pdf", {"query": " \n "}),
            ("retrieve_pdf", {"query": "x" * 2001}),
            ("document_status", None),
            ("document_status", []),
            ("document_status", {"extra": True}),
        ]
        for name, arguments in cases:
            with self.subTest(tool=name, arguments=arguments), \
                 patch.object(ask_agent, "retrieve_pdf_tool") as retrieve, \
                 patch.object(ask_agent, "document_status_tool") as status:
                with self.assertRaises(RuntimeError):
                    ask_agent.execute_tool(name, arguments, **TOOL_SETTINGS)
                retrieve.assert_not_called()
                status.assert_not_called()

    def test_executor_accepts_query_length_boundary_and_trims_it(self) -> None:
        query = "x" * 2000
        expected = ({"context": "evidence"}, [])
        with patch.object(ask_agent, "retrieve_pdf_tool", return_value=expected) as retrieve:
            actual = ask_agent.execute_tool("retrieve_pdf", {"query": f"  {query}  "}, **TOOL_SETTINGS)
        self.assertEqual(actual, expected)
        retrieve.assert_called_once_with(query=query, **TOOL_SETTINGS)

    def test_document_status_uses_temporary_metadata_and_ocr_review_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parsed_path = Path(directory) / "parsed.json"
            chunks_path = Path(directory) / "chunks.json"
            parsed_path.write_text(json.dumps({"pages": [
                {"page_number": 1, "needs_ocr": False},
                {"page_number": 2, "needs_ocr": True},
                {"page_number": 3, "needs_ocr": True},
            ]}), encoding="utf-8")
            chunks_path.write_text(json.dumps({
                "source": {"file_name": "synthetic.pdf"},
                "document": {"page_count": 3, "chunk_count": 4},
                "input": {"path": str(parsed_path)},
            }), encoding="utf-8")
            status, sources = ask_agent.execute_tool(
                "document_status", {}, **{**TOOL_SETTINGS, "chunks_json_path": chunks_path}
            )
            self.assertEqual(status["file_name"], "synthetic.pdf")
            self.assertEqual(status["page_count"], 3)
            self.assertEqual(status["chunk_count"], 4)
            self.assertEqual(status["ocr_review_pages"], [2, 3])
            self.assertIn("not implemented", status["limitations"])
            self.assertEqual(sources, [])
            parsed_path.unlink()
            unavailable = ask_agent.document_status_tool(chunks_path)
            self.assertIsNone(unavailable["ocr_review_pages"])

    def test_retrieval_tool_preserves_evidence_and_source_metadata(self) -> None:
        source = {"source_id": "S1", "chunk_id": "chunk-1", "page_range": [1, 2],
                  "rerank_score": 0.9, "text": "synthetic evidence"}
        context = {"context": "[S1] synthetic evidence", "sources": [source]}
        with patch.object(ask_agent, "build_context", return_value=context) as build:
            result, sources = ask_agent.retrieve_pdf_tool(query="fact?", **TOOL_SETTINGS)
        build.assert_called_once_with(query="fact?", **TOOL_SETTINGS)
        self.assertEqual(result["context"], context["context"])
        self.assertEqual(result["sources"], [{key: source[key] for key in (
            "source_id", "chunk_id", "page_range", "rerank_score"
        )}])
        self.assertEqual(sources, [source])


class AgentRunTests(unittest.TestCase):
    def test_each_approved_tool_runs_once_between_exactly_two_model_calls(self) -> None:
        for name in ("retrieve_pdf", "document_status"):
            with self.subTest(tool=name):
                self.check_bounded_run(name, empty=False)

    def test_empty_retrieval_evidence_still_has_a_single_answer_call(self) -> None:
        self.check_bounded_run("retrieve_pdf", empty=True)

    def check_bounded_run(self, name: str, empty: bool) -> None:
        source = {"source_id": "S1", "chunk_id": "chunk-1", "page_range": [1, 1], "rerank_score": 0.9}
        sources = [] if empty or name == "document_status" else [source]
        retrieved = {"context": "" if empty else "[S1] evidence", "sources": sources}
        metadata = {"file_name": "synthetic.pdf", "page_count": 1, "chunk_count": 1}
        responses = [tool_response(name, '{"query":"  fact?  "}' if name == "retrieve_pdf" else "{}"),
                     {"choices": [{"message": {"content": "Test answer."}}]}]
        calls: list[dict[str, Any]] = []

        def model(**kwargs: Any) -> dict[str, Any]:
            calls.append(copy.deepcopy(kwargs))
            return responses[len(calls) - 1]

        with patch.object(ask_agent, "call_deepseek", side_effect=model), \
             patch.object(ask_agent, "retrieve_pdf_tool", return_value=(retrieved, sources)) as retrieve, \
             patch.object(ask_agent, "document_status_tool", return_value=metadata) as status:
            result = ask_agent.run_agent(**RUN_SETTINGS)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(calls[0]["messages"]), 2)
        self.assertEqual(calls[0]["tool_choice"], "required")
        self.assertEqual({tool["function"]["name"] for tool in calls[0]["tools"]},
                         {"retrieve_pdf", "document_status"})
        self.assertNotIn("tools", calls[1])
        self.assertNotIn("tool_choice", calls[1])
        self.assertEqual([message["role"] for message in calls[1]["messages"]],
                         ["system", "user", "assistant", "tool"])
        tool_message = calls[1]["messages"][-1]
        self.assertEqual(tool_message["tool_call_id"], "call_1")
        self.assertEqual(json.loads(tool_message["content"]), retrieved if name == "retrieve_pdf" else metadata)
        if name == "retrieve_pdf":
            retrieve.assert_called_once_with(query="fact?", **TOOL_SETTINGS)
            status.assert_not_called()
        else:
            status.assert_called_once_with(TOOL_SETTINGS["chunks_json_path"])
            retrieve.assert_not_called()
        self.assertEqual(result["answer"], "Test answer.")
        self.assertEqual(result["sources"], sources)
        self.assertEqual(len(result["trace"]), 1)
        self.assertEqual(result["trace"][0]["tool"], name)
        self.assertEqual(result["trace"][0]["source_count"], len(sources))
        self.assertEqual(result["agent"]["max_tool_calls"], 1)

    def test_thinking_mode_is_rejected_before_model_or_tool_calls(self) -> None:
        with patch.object(ask_agent, "call_deepseek") as model, \
             patch.object(ask_agent, "execute_tool") as execute:
            with self.assertRaises(RuntimeError):
                ask_agent.run_agent(**{**RUN_SETTINGS, "thinking": "enabled"})
        model.assert_not_called()
        execute.assert_not_called()


class AgentCommandTests(unittest.TestCase):
    def test_cli_resolves_env_file_then_applies_explicit_config_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("DEEPSEEK_API_KEY=fixture-key\n"
                                "DEEPSEEK_BASE_URL=https://env.example.test\n"
                                "DEEPSEEK_MODEL=env-model\n", encoding="utf-8")
            for flags, expected_url, expected_model in [
                ([], "https://env.example.test", "env-model"),
                (["--base-url", "https://cli.example.test", "--llm-model", "cli-model"],
                 "https://cli.example.test", "cli-model"),
            ]:
                with self.subTest(flags=flags), patch.dict(os.environ, {}, clear=True), \
                     patch.object(sys, "argv", ["ask_agent.py", "unused.json", "question", "--env-file", str(env_path), *flags]), \
                     patch.object(ask_agent, "run_agent", return_value={}) as run, \
                     patch.object(ask_agent, "print_result"), redirect_stdout(io.StringIO()):
                    self.assertEqual(ask_agent.main(), 0)
                self.assertEqual(run.call_args.kwargs["api_key"], "fixture-key")
                self.assertEqual(run.call_args.kwargs["base_url"], expected_url)
                self.assertEqual(run.call_args.kwargs["llm_model"], expected_model)
                self.assertEqual(run.call_args.kwargs["thinking"], "disabled")

    def test_cli_rejects_enabled_thinking_before_running_agent(self) -> None:
        with patch.object(sys, "argv", ["ask_agent.py", "unused.json", "question", "--thinking", "enabled"]), \
             patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture-key"}, clear=True), \
             patch.object(ask_agent, "run_agent") as run, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                ask_agent.main()
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()


class ToolTransportTests(unittest.TestCase):
    def test_shared_transport_forwards_tools_only_when_requested(self) -> None:
        for planning in (True, False):
            with self.subTest(planning=planning), patch.object(answer_question.urllib.request, "urlopen") as open_url:
                open_url.return_value.__enter__.return_value.read.return_value = b'{"choices":[]}'
                extra = {"tools": ask_agent.TOOLS, "tool_choice": "required"} if planning else {}
                response = answer_question.call_deepseek(
                    api_key="test-key", base_url="https://example.test/", model="test-model",
                    messages=[{"role": "user", "content": "question"}], max_tokens=10,
                    temperature=0.2, thinking="disabled", reasoning_effort="high", timeout_seconds=5,
                    **extra,
                )
                request = open_url.call_args.args[0]
                payload = json.loads(request.data)
                self.assertEqual(request.full_url, "https://example.test/chat/completions")
                self.assertEqual(open_url.call_args.kwargs["timeout"], 5)
                self.assertEqual(response, {"choices": []})
                self.assertEqual(payload["thinking"], {"type": "disabled"})
                self.assertNotIn("reasoning_effort", payload)
                if planning:
                    self.assertEqual(payload["tools"], ask_agent.TOOLS)
                    self.assertEqual(payload["tool_choice"], "required")
                else:
                    self.assertNotIn("tools", payload)
                    self.assertNotIn("tool_choice", payload)


if __name__ == "__main__":
    unittest.main()
