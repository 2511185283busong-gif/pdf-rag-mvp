#!/usr/bin/env python3
"""Extract a local PDF into page-level JSON for a RAG MVP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pdfplumber


SCHEMA_VERSION = "0.4.0"

# Gaps wider than this fraction of the font size become word breaks. A fixed
# tolerance glues words together when a small font's space is narrower than it.
X_TOLERANCE_RATIO = 0.15
# Some fonts encode small capitals as private-use glyphs U+F761-U+F77A.
SMALL_CAPS_TO_ASCII = {
    code_point: chr(ord("a") + code_point - 0xF761)
    for code_point in range(0xF761, 0xF77B)
}
# pdfminer writes this placeholder for a glyph with no Unicode mapping. It carries
# no text, and on formula-heavy pages it can make up half of a chunk.
UNMAPPED_GLYPH = re.compile(r"\(cid:\d+\)")
# A line needs a word, a multi-digit number or an ideograph to count as content.
# Formula fragments such as "x = ∨x =" and subscript rows such as "1 2 1 2" have none.
LINE_CONTENT = re.compile(r"[^\W\d_]{2,}|\d{2,}|[\u4e00-\u9fff]")
# Single-digit table rows also lack content, but unlike subscripts they use body-size text.
NUMERIC_ROW = re.compile(r"\d+(?:\s+\d+)+")
BODY_TEXT_HEIGHT_RATIO = 0.8
# Some PDFs map common ideographs to look-alike radicals, such as U+2F3F for 手
# and U+2EDA for 页. They look right but do not match the real characters.
RADICALS_TO_IDEOGRAPHS = {
    **{
        code_point: unicodedata.normalize("NFKC", chr(code_point))
        for code_point in range(0x2F00, 0x2FD6)
    },
    **str.maketrans(
        "⺠⻄⻅⻆⻈⻉⻋⻒⻓⻔⻘⻙⻚⻛⻜⻝⻢⻣⻥⻬⻮⻰",
        "民西见角讠贝车长长门青韦页风飞食马骨鱼齐齿龙",
    ),
}
PARAGRAPH_GAP_RATIO = 1.5
HEADER_FOOTER_BAND = 0.15
REPEATED_LINE_MIN_PAGES = 3
REPEATED_LINE_MIN_SHARE = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = text.translate(SMALL_CAPS_TO_ASCII).translate(RADICALS_TO_IDEOGRAPHS)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_unmapped_glyphs(text: str) -> str:
    stripped = UNMAPPED_GLYPH.sub("", text)
    return " ".join(stripped.split()) if stripped != text else text


def extract_page_lines(page: Any) -> list[dict[str, Any]]:
    lines = page.extract_text_lines(
        x_tolerance_ratio=X_TOLERANCE_RATIO,
        return_chars=False,
    )
    extracted = []
    for line in lines:
        # Small capitals become letters here so the content check can read them.
        text = (
            strip_unmapped_glyphs(line["text"])
            .translate(SMALL_CAPS_TO_ASCII)
            .translate(RADICALS_TO_IDEOGRAPHS)
        )
        if text.strip():
            extracted.append(
                {"text": text, "top": float(line["top"]), "bottom": float(line["bottom"])}
            )
    return extracted


def count_unmapped_glyphs(page: Any) -> int:
    return sum(1 for char in page.chars if UNMAPPED_GLYPH.fullmatch(char["text"]))


def line_height(line: dict[str, Any]) -> float:
    return line["bottom"] - line["top"]


def split_low_information_lines(
    lines: list[dict[str, Any]],
    body_height: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def has_content(line: dict[str, Any]) -> bool:
        if LINE_CONTENT.search(line["text"]):
            return True
        return bool(NUMERIC_ROW.fullmatch(line["text"])) and (
            line_height(line) >= body_height * BODY_TEXT_HEIGHT_RATIO
        )

    kept = [line for line in lines if has_content(line)]
    removed = [line for line in lines if not has_content(line)]
    return kept, removed


def repeated_line_key(text: str) -> str:
    return re.sub(r"\d+", "#", " ".join(text.split())).casefold()


def is_in_header_footer_band(line: dict[str, Any], page_height: float) -> bool:
    band = page_height * HEADER_FOOTER_BAND
    return line["top"] <= band or line["bottom"] >= page_height - band


def find_repeated_lines(
    pages_lines: list[list[dict[str, Any]]],
    page_heights: list[float],
) -> set[str]:
    """Find headers and footers: lines repeating near the page edge across a document."""
    page_counts: dict[str, int] = {}
    for lines, height in zip(pages_lines, page_heights):
        keys = {
            repeated_line_key(line["text"])
            for line in lines
            if is_in_header_footer_band(line, height)
        }
        for key in keys:
            page_counts[key] = page_counts.get(key, 0) + 1

    min_pages = max(
        REPEATED_LINE_MIN_PAGES,
        math.ceil(len(pages_lines) * REPEATED_LINE_MIN_SHARE),
    )
    return {key for key, count in page_counts.items() if count >= min_pages}


def split_repeated_lines(
    lines: list[dict[str, Any]],
    page_height: float,
    repeated_keys: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    kept: list[dict[str, Any]] = []
    removed: list[str] = []
    for line in lines:
        if (
            is_in_header_footer_band(line, page_height)
            and repeated_line_key(line["text"]) in repeated_keys
        ):
            removed.append(line["text"])
        else:
            kept.append(line)
    return kept, removed


def lines_to_text(lines: list[dict[str, Any]]) -> str:
    """Join lines, marking unusually large vertical gaps as paragraph breaks."""
    if not lines:
        return ""

    steps = [current["top"] - previous["top"] for previous, current in zip(lines, lines[1:])]
    positive_steps = [step for step in steps if step > 0]
    typical_step = statistics.median(positive_steps) if len(positive_steps) >= 2 else None

    parts = [lines[0]["text"]]
    for step, line in zip(steps, lines[1:]):
        is_paragraph_break = typical_step is not None and step > typical_step * PARAGRAPH_GAP_RATIO
        parts.append("\n\n" if is_paragraph_break else "\n")
        parts.append(line["text"])
    return "".join(parts)


def count_words(text: str) -> int:
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    latin_words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text)
    return len(chinese_chars) + len(latin_words)


def parse_pdf(pdf_path: Path, min_text_chars: int) -> dict[str, Any]:
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file: {pdf_path}")

    pages: list[dict[str, Any]] = []
    warnings: list[str] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        pages_lines: list[list[dict[str, Any]]] = []
        extraction_errors: list[str | None] = []
        for page_index, page in enumerate(pdf.pages):
            extraction_error = None

            try:
                lines = extract_page_lines(page)
            except Exception as exc:  # Keep one bad page from failing the whole file.
                lines = []
                extraction_error = f"{type(exc).__name__}: {exc}"
                warnings.append(f"Page {page_index + 1}: {extraction_error}")

            pages_lines.append(lines)
            extraction_errors.append(extraction_error)

        page_heights = [float(page.height) for page in pdf.pages]
        repeated_keys = find_repeated_lines(pages_lines, page_heights)
        # Measured across the document: subscripts dominate a formula page's own median.
        body_height = statistics.median(
            [line_height(line) for lines in pages_lines for line in lines] or [0.0]
        )
        repeated_lines: dict[str, str] = {}

        for page_index, page in enumerate(pdf.pages):
            lines = pages_lines[page_index]
            # OCR review looks at everything extracted, including headers and footers.
            needs_ocr = len(normalize_text(lines_to_text(lines))) < min_text_chars
            kept_lines, removed_lines = split_repeated_lines(
                lines, page_heights[page_index], repeated_keys
            )
            kept_lines, low_information_lines = split_low_information_lines(
                kept_lines, body_height
            )
            removed_lines = [normalize_text(line) for line in removed_lines]
            for line in removed_lines:
                repeated_lines.setdefault(repeated_line_key(line), line)
            text = normalize_text(lines_to_text(kept_lines))

            pages.append(
                {
                    "page_index": page_index,
                    "page_number": page_index + 1,
                    "width": round(float(page.width), 2),
                    "height": round(float(page.height), 2),
                    "char_count": len(text),
                    "word_count": count_words(text),
                    "has_text": bool(text),
                    "needs_ocr": needs_ocr,
                    "text": text,
                    "removed_lines": removed_lines,
                    "low_information_lines_removed": len(low_information_lines),
                    "unmapped_glyphs_removed": count_unmapped_glyphs(page),
                    "extraction_error": extraction_errors[page_index],
                }
            )

        metadata = dict(pdf.metadata or {})

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(pdf_path),
            "file_name": pdf_path.name,
            "size_bytes": pdf_path.stat().st_size,
            "sha256": sha256_file(pdf_path),
        },
        "parser": {
            "name": "pdfplumber",
            "version": getattr(pdfplumber, "__version__", "unknown"),
            "min_text_chars_for_ocr_flag": min_text_chars,
            "x_tolerance_ratio": X_TOLERANCE_RATIO,
        },
        "document": {
            "page_count": len(pages),
            "metadata": metadata,
            "repeated_lines_removed": sorted(repeated_lines.values()),
            "low_information_lines_removed": sum(
                page["low_information_lines_removed"] for page in pages
            ),
            "unmapped_glyphs_removed": sum(page["unmapped_glyphs_removed"] for page in pages),
        },
        "pages": pages,
        "warnings": warnings,
    }


def default_output_path(pdf_path: Path) -> Path:
    return Path("parsed") / f"{pdf_path.stem}.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract a PDF into page-level JSON with page numbers."
    )
    parser.add_argument("pdf", type=Path, help="Path to the local PDF file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON path. Defaults to parsed/<pdf-name>.json.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=20,
        help="Pages with fewer characters are marked as needs_ocr. Default: 20.",
    )

    args = parser.parse_args()
    output_path = args.output or default_output_path(args.pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = parse_pdf(args.pdf, args.min_text_chars)
    except Exception as exc:
        print(f"Failed to parse PDF: {exc}", file=sys.stderr)
        return 1

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    page_count = result["document"]["page_count"]
    ocr_count = sum(1 for page in result["pages"] if page["needs_ocr"])
    print(f"Wrote {output_path}")
    print(f"Pages: {page_count}; pages needing OCR review: {ocr_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
