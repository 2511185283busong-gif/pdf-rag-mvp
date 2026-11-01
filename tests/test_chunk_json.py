from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from chunk_json import (  # noqa: E402
    chunk_units,
    split_by_size,
    split_long_text,
    split_page_into_units,
    text_tail,
)


class SemanticUnitTests(unittest.TestCase):
    def units(self, text: str) -> list[str]:
        page = {"page_number": 1, "text": text}
        return [unit["text"] for unit in split_page_into_units(page)]

    def test_blank_lines_separate_paragraphs(self) -> None:
        self.assertEqual(
            self.units("First line\ncontinues\n\nSecond paragraph"),
            ["First line continues", "Second paragraph"],
        )

    def test_bullets_start_new_units(self) -> None:
        self.assertEqual(
            self.units("Intro\n• first\nwraps\n- second"),
            ["Intro", "• first wraps", "- second"],
        )

    def test_chinese_headings_start_new_units(self) -> None:
        self.assertEqual(
            self.units("第一章 引言\n正文一\n二、方法\n正文二\n（三）结果\n3、讨论"),
            ["第一章引言正文一", "二、方法正文二", "（三）结果", "3、讨论"],
        )

    def test_chinese_lines_join_without_spaces(self) -> None:
        self.assertEqual(self.units("这是第一行\n这是第 二行"), ["这是第一行这是第二行"])

    def test_latin_lines_still_join_with_spaces(self) -> None:
        self.assertEqual(self.units("hello\nworld"), ["hello world"])


class LongTextTests(unittest.TestCase):
    def test_chinese_text_splits_at_sentence_boundaries(self) -> None:
        sentence = "这是一个没有空格的中文句子。"
        parts = split_long_text(sentence * 20, max_chars=100)

        self.assertGreater(len(parts), 1)
        self.assertEqual("".join(parts), sentence * 20)
        self.assertTrue(all(len(part) <= 100 and part.endswith("。") for part in parts))

    def test_closing_quote_stays_with_its_sentence(self) -> None:
        parts = split_long_text("他说：“第一句。”第二句。", max_chars=9)

        self.assertEqual(parts[-1], "第二句。")

    def test_oversized_chinese_sentence_splits_at_clauses(self) -> None:
        clause = "这是一个很长的分句，"
        parts = split_long_text(clause * 10, max_chars=30)

        self.assertEqual("".join(parts), clause * 10)
        self.assertTrue(all(part.endswith("，") for part in parts))

    def test_ideographs_count_double_towards_the_limit(self) -> None:
        sentence = "这是十个汉字的句子呀。"
        parts = split_long_text(sentence * 4, max_chars=42)

        self.assertEqual(parts, [sentence * 2, sentence * 2])

    def test_unbroken_chinese_text_respects_the_limit(self) -> None:
        parts = split_long_text("汉" * 50, max_chars=20)

        self.assertEqual("".join(parts), "汉" * 50)
        self.assertTrue(all(len(part) <= 10 for part in parts))

    def test_overlap_starts_at_a_whole_word(self) -> None:
        self.assertEqual(text_tail("alpha bravo charlie delta", 13), "charlie delta")
        self.assertEqual(text_tail("alpha bravo charlie delta", 15), "charlie delta")

    def test_english_sentences_keep_spaces(self) -> None:
        self.assertEqual(
            split_long_text("One two. Three four. Five six.", max_chars=20),
            ["One two. Three four.", "Five six."],
        )


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

    def test_overlap_never_pushes_a_chunk_past_max_chars(self) -> None:
        units = [
            {"text": "a" * 15, "source_pages": [1]},
            {"text": "b" * 20, "source_pages": [2]},
            {"text": "c" * 20, "source_pages": [3]},
        ]

        chunks = chunk_units(
            units=units,
            document_id="example",
            target_chars=18,
            max_chars=20,
            overlap_chars=5,
        )

        self.assertEqual([chunk["text"] for chunk in chunks], ["a" * 15, "b" * 20, "c" * 20])
        self.assertEqual(chunks[1]["overlap_from_previous"], "")
        self.assertEqual(chunks[1]["page_range"], "2")

    def test_page_segments_keep_text_with_its_page(self) -> None:
        units = [
            {"text": "a" * 10, "source_pages": [1]},
            {"text": "b" * 10, "source_pages": [1]},
            {"text": "c" * 10, "source_pages": [2]},
        ]

        chunk = chunk_units(
            units=units,
            document_id="example",
            target_chars=100,
            max_chars=200,
            overlap_chars=0,
        )[0]

        self.assertEqual(
            chunk["page_segments"],
            [
                {"page": 1, "text": f"{'a' * 10}\n\n{'b' * 10}"},
                {"page": 2, "text": "c" * 10},
            ],
        )

    def test_unbroken_token_respects_max_chars(self) -> None:
        text = "a" * 2500

        parts = split_by_size(text, max_chars=1100)

        self.assertEqual("".join(parts), text)
        self.assertTrue(all(len(part) <= 1100 for part in parts))


if __name__ == "__main__":
    unittest.main()
