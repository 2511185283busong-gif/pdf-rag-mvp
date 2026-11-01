from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from parse_pdf import (  # noqa: E402
    extract_page_lines,
    find_repeated_lines,
    lines_to_text,
    normalize_text,
    split_repeated_lines,
)


def line(text: str, top: float) -> dict:
    return {"text": text, "top": top, "bottom": top + 10}


class ExtractionTests(unittest.TestCase):
    def test_tolerance_scales_with_font_size(self) -> None:
        captured: dict = {}

        class FakePage:
            def extract_text_lines(self, **kwargs: object) -> list[dict]:
                captured.update(kwargs)
                return [{"text": "kept", "top": 1, "bottom": 2}, {"text": "  ", "top": 3, "bottom": 4}]

        lines = extract_page_lines(FakePage())

        self.assertIn("x_tolerance_ratio", captured)
        self.assertEqual([item["text"] for item in lines], ["kept"])

    def test_small_cap_glyphs_become_ascii(self) -> None:
        self.assertEqual(normalize_text("?(j, x)"), "ok?(j, x)")


class RepeatedLineTests(unittest.TestCase):
    def test_edge_lines_with_changing_page_numbers_are_removed(self) -> None:
        pages = [
            [line("Course Title", 5), line(f"Body {page}", 300), line(f"Page {page}", 780)]
            for page in range(1, 5)
        ]
        heights = [800.0] * 4

        keys = find_repeated_lines(pages, heights)
        kept, removed = split_repeated_lines(pages[1], 800.0, keys)

        self.assertEqual([item["text"] for item in kept], ["Body 2"])
        self.assertEqual(removed, ["Course Title", "Page 2"])

    def test_repeated_body_lines_are_kept(self) -> None:
        pages = [[line("Same sentence in the body", 400)] for _ in range(4)]

        self.assertEqual(find_repeated_lines(pages, [800.0] * 4), set())

    def test_short_documents_keep_their_lines(self) -> None:
        pages = [[line("Name", 5)], [line("Name", 5)]]

        self.assertEqual(find_repeated_lines(pages, [800.0, 800.0]), set())


class ParagraphTests(unittest.TestCase):
    def test_large_vertical_gap_becomes_blank_line(self) -> None:
        lines = [line("one", 100), line("two", 118), line("three", 136), line("four", 190)]

        self.assertEqual(lines_to_text(lines), "one\ntwo\nthree\n\nfour")

    def test_too_few_lines_have_no_paragraph_breaks(self) -> None:
        self.assertEqual(lines_to_text([line("one", 100), line("two", 300)]), "one\ntwo")


if __name__ == "__main__":
    unittest.main()
