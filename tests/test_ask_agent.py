from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from ask_agent import document_status_tool, parse_tool_call  # noqa: E402


def tool_response(name: str, arguments: str) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            }
        ]
    }


class AgentToolValidationTests(unittest.TestCase):
    def test_parse_allowed_retrieve_call(self) -> None:
        parsed = parse_tool_call(tool_response("retrieve_pdf", '{"query":"ABT"}'))

        self.assertEqual(parsed["name"], "retrieve_pdf")
        self.assertEqual(parsed["arguments"], {"query": "ABT"})

    def test_reject_unknown_tool(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not allowed"):
            parse_tool_call(tool_response("run_shell", "{}"))

    def test_document_status_reads_chunk_metadata(self) -> None:
        status = document_status_tool(Path("chunks/resume2_chunks.json"))

        self.assertEqual(status["chunk_count"], 3)
        self.assertEqual(status["page_count"], 2)


if __name__ == "__main__":
    unittest.main()
