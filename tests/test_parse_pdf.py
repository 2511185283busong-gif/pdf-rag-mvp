from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from parse_pdf import (  # noqa: E402
    count_unmapped_glyphs,
    extract_page_lines,
    find_repeated_lines,
    lines_to_text,
    normalize_text,
    split_low_information_lines,
    split_repeated_lines,
    strip_unmapped_glyphs,
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


class NoiseTests(unittest.TestCase):
    def test_unmapped_glyphs_are_stripped(self) -> None:
        class FakePage:
            def extract_text_lines(self, **kwargs: object) -> list[dict]:
                return [
                    {"text": "x = (cid:8) (cid:8) end", "top": 1, "bottom": 2},
                    {"text": "(cid:8) (cid:8)", "top": 3, "bottom": 4},
                ]

        lines = extract_page_lines(FakePage())

        self.assertEqual(strip_unmapped_glyphs("keeps  its spacing"), "keeps  its spacing")
        self.assertEqual([item["text"] for item in lines], ["x = end"])

    def test_unmapped_glyphs_are_counted_from_page_characters(self) -> None:
        class FakePage:
            chars = [{"text": "(cid:8)"}, {"text": "x"}, {"text": "(cid:12)"}]

        self.assertEqual(count_unmapped_glyphs(FakePage()), 2)

    def test_small_cap_lines_count_as_content(self) -> None:
        class FakePage:
            def extract_text_lines(self, **kwargs: object) -> list[dict]:
                return [{"text": "\uf772\uf765\uf776(x, y)", "top": 1, "bottom": 2}]

        kept, removed = split_low_information_lines(extract_page_lines(FakePage()))

        self.assertEqual([item["text"] for item in kept], ["rev(x, y)"])
        self.assertEqual(removed, [])

    def test_lines_without_content_are_separated(self) -> None:
        contentful = ["ok?(x ,,0) =", "Year 2024", "12 34 56", "第一", "是", "2 then return"]
        fragments = ["x = ∨x =", "1 2 1 2 1 2", "¬(x = ∧x = )", "n", ")"]
        lines = [line(text, 100) for text in contentful + fragments]

        kept, removed = split_low_information_lines(lines)

        self.assertEqual([item["text"] for item in kept], contentful)
        self.assertEqual([item["text"] for item in removed], fragments)


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
