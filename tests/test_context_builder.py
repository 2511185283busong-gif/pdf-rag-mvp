from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from context_builder import build_context, trim_text  # noqa: E402


class ContextBuilderTests(unittest.TestCase):
    def test_trim_text_includes_ellipsis_inside_limit(self) -> None:
        text, truncated = trim_text("abcdefghij", max_chars=5)

        self.assertTrue(truncated)
        self.assertEqual(text, "ab...")
        self.assertEqual(len(text), 5)

    @patch("context_builder.rerank_search")
    def test_context_including_headers_stays_within_budget(self, rerank_search) -> None:
        chunks = {
            "source": {},
            "chunks": [
                {
                    "chunk_id": "doc_chunk_0000",
                    "chunk_index": 0,
                    "text": "a" * 300,
                },
                {
                    "chunk_id": "doc_chunk_0001",
                    "chunk_index": 1,
                    "text": "b" * 300,
                },
            ],
        }
        rerank_search.return_value = {
            "results": [
                {
                    "chunk_id": "doc_chunk_0000",
                    "chunk_index": 0,
                    "page_range": "1",
                    "source_pages": [1],
                    "rerank_rank": 1,
                    "rerank_score": 1.0,
                    "base_rank": 1,
                    "base_similarity": 0.9,
                },
                {
                    "chunk_id": "doc_chunk_0001",
                    "chunk_index": 1,
                    "page_range": "2",
                    "source_pages": [2],
                    "rerank_rank": 2,
                    "rerank_score": 0.5,
                    "base_rank": 2,
                    "base_similarity": 0.8,
                },
            ],
            "method": {},
            "input": {},
        }

        with tempfile.TemporaryDirectory() as directory:
            chunks_path = Path(directory) / "chunks.json"
            chunks_path.write_text(json.dumps(chunks), encoding="utf-8")
            result = build_context(
                chunks_json_path=chunks_path,
                query="question",
                top_k=2,
                candidate_k=2,
                embedding_model="embedding",
                rerank_model="reranker",
                max_context_chars=160,
                max_chunk_chars=120,
            )

        self.assertLessEqual(len(result["context"]), 160)
        self.assertEqual(result["budget"]["used_context_chars"], len(result["context"]))
        self.assertTrue(
            all(source["included_chars"] <= 120 for source in result["sources"])
        )


if __name__ == "__main__":
    unittest.main()
