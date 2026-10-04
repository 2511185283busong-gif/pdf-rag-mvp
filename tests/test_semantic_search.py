from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from semantic_search import semantic_search  # noqa: E402


class FakeSentenceTransformer:
    local_modes: list[bool] = []

    def __init__(self, model_name, cache_folder, local_files_only):
        self.local_modes.append(local_files_only)

    def encode(self, texts, normalize_embeddings, show_progress_bar):
        if isinstance(texts, str):
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class SemanticSearchTests(unittest.TestCase):
    def test_local_mode_does_not_stick_in_process_environment(self) -> None:
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = FakeSentenceTransformer
        chunks = {
            "source": {},
            "chunks": [
                {
                    "chunk_id": "doc_chunk_0000",
                    "chunk_index": 0,
                    "text": "text",
                    "page_range": "1",
                    "source_pages": [1],
                }
            ],
        }
        FakeSentenceTransformer.local_modes = []

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            model_path.mkdir()
            chunks_path = Path(directory) / "chunks.json"
            chunks_path.write_text(json.dumps(chunks), encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
                    semantic_search(chunks_path, "query", 1, str(model_path), 50, True)
                    semantic_search(chunks_path, "query", 1, str(model_path), 50, False)

                self.assertNotIn("HF_HUB_OFFLINE", os.environ)
                self.assertNotIn("TRANSFORMERS_OFFLINE", os.environ)

        self.assertEqual(FakeSentenceTransformer.local_modes, [True, False])


if __name__ == "__main__":
    unittest.main()
