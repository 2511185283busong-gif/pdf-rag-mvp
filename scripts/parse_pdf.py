#!/usr/bin/env python3
"""Extract a local PDF into page-level JSON for a RAG MVP."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pdfplumber


SCHEMA_VERSION = "0.1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1
            extraction_error = None

            try:
                raw_text = page.extract_text() or ""
            except Exception as exc:  # Keep one bad page from failing the whole file.
                raw_text = ""
                extraction_error = f"{type(exc).__name__}: {exc}"

            text = normalize_text(raw_text)
            char_count = len(text)
            needs_ocr = char_count < min_text_chars

            if extraction_error:
                warnings.append(f"Page {page_number}: {extraction_error}")

            pages.append(
                {
                    "page_index": page_index,
                    "page_number": page_number,
                    "width": round(float(page.width), 2),
                    "height": round(float(page.height), 2),
                    "char_count": char_count,
                    "word_count": count_words(text),
                    "has_text": char_count > 0,
                    "needs_ocr": needs_ocr,
                    "text": text,
                    "extraction_error": extraction_error,
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
        },
        "document": {
            "page_count": len(pages),
            "metadata": metadata,
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
