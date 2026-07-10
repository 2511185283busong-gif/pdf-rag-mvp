from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from chunk_json import chunk_units  # noqa: E402


class ChunkOverlapProvenanceTests(unittest.TestCase):
    def test_overlap_keeps_only_the_tail_pages(self) -> None:
        units = [
            {"text": "a" * 10, "source_pages": [1]},
            {"text": "b" * 10, "source_pages": [2]},
            {"text": "c" * 10, "source_pages": [3]},
        ]

        chunks = chunk_units(
            units=units,
            document_id="example",
            target_chars=10,
            max_chars=20,
            overlap_chars=5,
        )

        self.assertEqual([chunk["page_range"] for chunk in chunks], ["1", "1-2", "2-3"])


if __name__ == "__main__":
    unittest.main()
