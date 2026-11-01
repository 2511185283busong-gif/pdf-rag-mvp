#!/usr/bin/env python3
"""Build semantic-ish chunks from parsed PDF JSON while preserving page metadata."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.2.0"
DEFAULT_TARGET_CHARS = 800
DEFAULT_MAX_CHARS = 1100
DEFAULT_OVERLAP_CHARS = 120
CJK_IDEOGRAPHS = "㐀-鿿豈-﫿"
CJK_TEXT = f"{CJK_IDEOGRAPHS}　-〿＀-￯"
CJK_NUMERALS = "一二三四五六七八九十百千零〇"
CJK_HEADING_PATTERN = (
    f"^(第[{CJK_NUMERALS}\\d]+[章节部分篇条讲]"
    f"|[{CJK_NUMERALS}]+[、．]"
    f"|[（(][{CJK_NUMERALS}\\d]+[）)]"
    r"|\d+(\.\d+)*[、．])"
)


def normalize_inline_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def normalize_unit_text(text: str) -> str:
    text = normalize_inline_text(text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(f"(?<=[{CJK_IDEOGRAPHS}]) (?=[{CJK_IDEOGRAPHS}])", "", text)
    return text.strip()


def text_size(text: str) -> int:
    """Measure text for chunk limits; an ideograph costs about two Latin characters in tokens."""
    return len(text) + len(re.findall(f"[{CJK_IDEOGRAPHS}]", text))


def count_words(text: str) -> int:
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    latin_words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text)
    return len(chinese_chars) + len(latin_words)


def unique_sorted_pages(pages: list[int]) -> list[int]:
    return sorted(set(pages))


def page_range_text(pages: list[int]) -> str:
    pages = unique_sorted_pages(pages)
    if not pages:
        return ""
    ranges: list[str] = []
    start = pages[0]
    prev = pages[0]

    for page in pages[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = page

    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def starts_new_unit(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^(\d+(\.\d+)*|[A-Z])[\).]\s+\S+", stripped):
        return True
    if re.match(CJK_HEADING_PATTERN, stripped):
        return True
    if re.match(r"^[•·▪◦●○■□‣*–—-]\s+\S", stripped):
        return True
    if stripped.endswith(":") and len(stripped) <= 90:
        return True
    if stripped.endswith("：") and len(stripped) <= 40:
        return True
    if re.match(r"^(Dear\s+\w+|Best regards,?)$", stripped, re.IGNORECASE):
        return True
    return False


def is_cjk_boundary(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    return bool(
        re.match(f"[{CJK_TEXT}]", previous[-1]) or re.match(f"[{CJK_TEXT}]", current[0])
    )


def join_text(previous: str, current: str) -> str:
    separator = "" if is_cjk_boundary(previous, current) else " "
    return f"{previous}{separator}{current}"


def join_line(previous: str, current: str) -> str:
    previous = previous.rstrip()
    current = current.strip()
    if previous.endswith("-") and current and current[0].islower():
        return f"{previous[:-1]}{current}"
    return join_text(previous, current)


def split_paragraph_into_units(paragraph: str, page_number: int) -> list[dict[str, Any]]:
    text = normalize_inline_text(paragraph)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    units: list[dict[str, Any]] = []
    current = ""

    for line in lines:
        if current and starts_new_unit(line):
            unit_text = normalize_unit_text(current)
            if unit_text:
                units.append(
                    {
                        "text": unit_text,
                        "source_pages": [page_number],
                    }
                )
            current = line
        else:
            current = line if not current else join_line(current, line)

    unit_text = normalize_unit_text(current)
    if unit_text:
        units.append(
            {
                "text": unit_text,
                "source_pages": [page_number],
            }
        )

    return units


def split_page_into_units(page: dict[str, Any]) -> list[dict[str, Any]]:
    page_number = int(page["page_number"])
    text = page.get("text", "").replace("\r\n", "\n").replace("\r", "\n")
    units: list[dict[str, Any]] = []

    for paragraph in re.split(r"\n\s*\n", text):
        units.extend(split_paragraph_into_units(paragraph, page_number))

    return units


def build_semantic_units(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []

    for page in pages:
        units.extend(split_page_into_units(page))

    return units


def split_sentences(text: str) -> list[str]:
    marked = re.sub(r"([。！？；][”’」』）)]*)\s*", "\\1\n", text)
    marked = re.sub(r"(?<=[.!?;])\s+", "\n", marked)
    return [part.strip() for part in marked.split("\n") if part.strip()]


def split_oversized_sentence(sentence: str, max_chars: int) -> list[str]:
    marked = re.sub(r"([，、：])\s*", "\\1\n", sentence)
    marked = re.sub(r"(?<=[,:])\s+", "\n", marked)
    pieces: list[str] = []
    for clause in marked.split("\n"):
        clause = clause.strip()
        if text_size(clause) > max_chars:
            pieces.extend(split_by_size(clause, max_chars))
        elif clause:
            pieces.append(clause)
    return pieces


def split_long_text(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    current = ""

    for sentence in split_sentences(text):
        pieces = (
            [sentence]
            if text_size(sentence) <= max_chars
            else split_oversized_sentence(sentence, max_chars)
        )
        for piece in pieces:
            candidate = piece if not current else join_text(current, piece)
            if text_size(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                parts.append(current)
            current = piece

    if current:
        parts.append(current)

    return parts


def split_by_size(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    parts: list[str] = []
    current = ""

    for word in words:
        if text_size(word) > max_chars:
            if current:
                parts.append(current)
                current = ""
            # Ideographs count double, so halve the slice length for them.
            slice_chars = max(max_chars // 2, 1) if text_size(word) > len(word) else max_chars
            while text_size(word) > max_chars:
                parts.append(word[:slice_chars])
                word = word[slice_chars:]
            current = word
            continue

        candidate = word if not current else f"{current} {word}"
        if text_size(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            parts.append(current)
            current = word
        else:
            current = word

    if current:
        parts.append(current)

    return parts


def expand_oversized_units(
    units: list[dict[str, Any]], max_chars: int
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []

    for unit in units:
        text = unit["text"]
        if text_size(text) <= max_chars:
            expanded.append(unit)
            continue

        for part in split_long_text(text, max_chars):
            expanded.append(
                {
                    "text": part,
                    "source_pages": unit["source_pages"],
                }
            )

    return expanded


def text_tail(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return ""
    tail = text[-overlap_chars:].strip()
    boundary = max(tail.find(". "), tail.find("。"), tail.find("\n"))
    if boundary > 0 and boundary + 1 < len(tail):
        return tail[boundary + 1 :].strip()
    # The cut usually lands inside a word, so start at the next whole one.
    space = tail.find(" ")
    cut_inside_word = not text[-overlap_chars - 1].isspace()
    if cut_inside_word and space > 0 and space + 1 < len(tail):
        return tail[space + 1 :].strip()
    return tail


def make_chunk(
    chunk_index: int,
    units: list[dict[str, Any]],
    document_id: str,
    overlap_from_previous: str | None = None,
) -> dict[str, Any]:
    text = "\n\n".join(unit["text"] for unit in units).strip()
    source_pages = unique_sorted_pages(
        page for unit in units for page in unit["source_pages"]
    )

    return {
        "chunk_id": f"{document_id}_chunk_{chunk_index:04d}",
        "chunk_index": chunk_index,
        "text": text,
        "char_count": len(text),
        "word_count": count_words(text),
        "source_pages": source_pages,
        "page_range": page_range_text(source_pages),
        "start_page": source_pages[0] if source_pages else None,
        "end_page": source_pages[-1] if source_pages else None,
        "overlap_from_previous": overlap_from_previous or "",
    }


def has_content_unit(units: list[dict[str, Any]]) -> bool:
    return any(not unit.get("is_overlap") for unit in units)


def tail_source_pages(units: list[dict[str, Any]]) -> list[int]:
    """The overlap tail always ends in the last non-overlap semantic unit."""
    for unit in reversed(units):
        if not unit.get("is_overlap"):
            return unit["source_pages"]
    return []


def overlap_units(text: str, source_pages: list[int]) -> list[dict[str, Any]]:
    if not text:
        return []
    return [
        {
            "text": text,
            "source_pages": source_pages,
            "is_overlap": True,
        }
    ]


def chunk_units(
    units: list[dict[str, Any]],
    document_id: str,
    target_chars: int,
    max_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current_units: list[dict[str, Any]] = []
    current_len = 0
    previous_tail = ""
    previous_tail_pages: list[int] = []

    for unit in expand_oversized_units(units, max_chars):
        unit_len = text_size(unit["text"])
        separator_len = 2 if current_units else 0
        candidate_len = current_len + separator_len + unit_len

        if current_units and candidate_len > max_chars:
            if has_content_unit(current_units):
                chunk = make_chunk(len(chunks), current_units, document_id, previous_tail)
                chunks.append(chunk)
                previous_tail = text_tail(chunk["text"], overlap_chars)
                previous_tail_pages = tail_source_pages(current_units)
                current_units = overlap_units(previous_tail, previous_tail_pages)
                current_len = text_size(previous_tail)

            # Drop the overlap when it would push this unit past the hard limit.
            if not has_content_unit(current_units) and current_len + 2 + unit_len > max_chars:
                previous_tail = ""
                current_units = []
                current_len = 0

        current_units.append(unit)
        current_len += (2 if current_len else 0) + unit_len

        if current_len >= target_chars and has_content_unit(current_units):
            chunk = make_chunk(len(chunks), current_units, document_id, previous_tail)
            chunks.append(chunk)
            previous_tail = text_tail(chunk["text"], overlap_chars)
            previous_tail_pages = tail_source_pages(current_units)
            current_units = overlap_units(previous_tail, previous_tail_pages)
            current_len = text_size(previous_tail)

    if current_units and has_content_unit(current_units):
        chunks.append(make_chunk(len(chunks), current_units, document_id, previous_tail))

    return chunks


def default_output_path(input_path: Path) -> Path:
    return Path("chunks") / f"{input_path.stem.replace('_pages', '')}_chunks.json"


def build_chunks(
    parsed_json_path: Path,
    target_chars: int,
    max_chars: int,
    overlap_chars: int,
) -> dict[str, Any]:
    parsed = json.loads(parsed_json_path.read_text(encoding="utf-8"))
    source = parsed.get("source", {})
    document_id = Path(source.get("file_name") or parsed_json_path.stem).stem
    pages = parsed.get("pages", [])

    semantic_units = build_semantic_units(pages)
    chunks = chunk_units(
        semantic_units,
        document_id,
        target_chars,
        max_chars,
        overlap_chars,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "input": {
            "path": str(parsed_json_path.resolve()),
            "schema_version": parsed.get("schema_version"),
        },
        "chunker": {
            "name": "rule_based_semantic_chunker",
            "version": SCHEMA_VERSION,
            "target_chars": target_chars,
            "max_chars": max_chars,
            "overlap_chars": overlap_chars,
            "size_unit": "characters, with CJK ideographs counted twice",
            "notes": "Pages are metadata only. Page changes do not force chunk boundaries.",
        },
        "document": {
            "page_count": parsed.get("document", {}).get("page_count", len(pages)),
            "semantic_unit_count": len(semantic_units),
            "chunk_count": len(chunks),
        },
        "chunks": chunks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk parsed PDF JSON while preserving source page metadata."
    )
    parser.add_argument(
        "parsed_json",
        type=Path,
        help="Path to parsed PDF JSON, for example parsed/resume2.json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON path. Defaults to chunks/<input-name>_chunks.json.",
    )
    parser.add_argument(
        "--target-chars",
        type=int,
        default=DEFAULT_TARGET_CHARS,
        help=f"Preferred chunk size. Default: {DEFAULT_TARGET_CHARS}.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"Hard chunk size limit before splitting. Default: {DEFAULT_MAX_CHARS}.",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=DEFAULT_OVERLAP_CHARS,
        help=f"Metadata-only overlap preview size. Default: {DEFAULT_OVERLAP_CHARS}.",
    )

    args = parser.parse_args()
    output_path = args.output or default_output_path(args.parsed_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.target_chars <= 0 or args.max_chars <= 0:
        print("target-chars and max-chars must be positive.", file=sys.stderr)
        return 1
    if args.target_chars > args.max_chars:
        print("target-chars must be less than or equal to max-chars.", file=sys.stderr)
        return 1

    try:
        result = build_chunks(
            args.parsed_json,
            args.target_chars,
            args.max_chars,
            args.overlap_chars,
        )
    except Exception as exc:
        print(f"Failed to build chunks: {exc}", file=sys.stderr)
        return 1

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    document = result["document"]
    print(f"Wrote {output_path}")
    print(
        "Semantic units: "
        f"{document['semantic_unit_count']}; chunks: {document['chunk_count']}"
    )
    for chunk in result["chunks"]:
        print(
            f"{chunk['chunk_id']}: pages={chunk['page_range']} "
            f"chars={chunk['char_count']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
