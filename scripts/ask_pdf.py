#!/usr/bin/env python3
"""Run the complete PDF RAG pipeline from one command."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from answer_question import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    answer_question,
    load_env_file,
    print_answer,
)
from chunk_json import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    DEFAULT_TARGET_CHARS,
    build_chunks,
)
from parse_pdf import parse_pdf
from rerank_search import DEFAULT_RERANK_MODEL
from semantic_search import DEFAULT_MODEL


def output_paths(pdf_path: Path, parsed_dir: Path, chunks_dir: Path) -> tuple[Path, Path]:
    return (
        parsed_dir / f"{pdf_path.stem}.json",
        chunks_dir / f"{pdf_path.stem}_chunks.json",
    )


def run_pipeline(
    pdf_path: Path,
    query: str,
    parsed_dir: Path,
    chunks_dir: Path,
    min_text_chars: int,
    target_chars: int,
    max_chars: int,
    overlap_chars: int,
    top_k: int,
    candidate_k: int,
    embedding_model: str,
    rerank_model: str,
    max_context_chars: int,
    max_chunk_chars: int,
    local_files_only: bool,
    api_key: str,
    base_url: str,
    llm_model: str,
    max_tokens: int,
    temperature: float,
    thinking: str,
    reasoning_effort: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    parsed_path, chunks_path = output_paths(pdf_path, parsed_dir, chunks_dir)
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    parsed = parse_pdf(pdf_path, min_text_chars)
    parsed_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

    chunked = build_chunks(parsed_path, target_chars, max_chars, overlap_chars)
    chunks_path.write_text(json.dumps(chunked, ensure_ascii=False, indent=2), encoding="utf-8")

    answer = answer_question(
        chunks_json_path=chunks_path,
        query=query,
        top_k=top_k,
        candidate_k=candidate_k,
        embedding_model=embedding_model,
        rerank_model=rerank_model,
        max_context_chars=max_context_chars,
        max_chunk_chars=max_chunk_chars,
        local_files_only=local_files_only,
        api_key=api_key,
        base_url=base_url,
        llm_model=llm_model,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout_seconds,
    )

    return {
        "input_pdf": str(pdf_path.resolve()),
        "parsed_json": str(parsed_path.resolve()),
        "chunks_json": str(chunks_path.resolve()),
        "parsing": {
            "page_count": parsed["document"]["page_count"],
            "pages_needing_ocr_review": sum(
                1 for page in parsed["pages"] if page["needs_ocr"]
            ),
        },
        "chunking": chunked["document"],
        "answer": answer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse a PDF, build chunks, retrieve, rerank, and answer with DeepSeek."
    )
    parser.add_argument("pdf", type=Path, help="Path to the source PDF.")
    parser.add_argument("query", nargs="+", help="Question to answer.")
    parser.add_argument("--parsed-dir", type=Path, default=Path("parsed"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("chunks"))
    parser.add_argument("--min-text-chars", type=int, default=20)
    parser.add_argument("--target-chars", type=int, default=DEFAULT_TARGET_CHARS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    parser.add_argument("--top-k", type=int, default=2, help="Sources included in context.")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--max-context-chars", type=int, default=3000)
    parser.add_argument("--max-chunk-chars", type=int, default=1200)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
    )
    parser.add_argument(
        "--llm-model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    )
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--thinking", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--reasoning-effort", choices=["high", "max"], default="high")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    args = parser.parse_args()

    query = " ".join(args.query).strip()
    load_env_file(args.env_file)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()

    if not query:
        print("query cannot be empty.", file=sys.stderr)
        return 1
    if not api_key:
        print("DEEPSEEK_API_KEY is not set. Put it in .env.", file=sys.stderr)
        return 1
    if not args.pdf.exists() or args.pdf.suffix.lower() != ".pdf":
        print(f"Expected an existing PDF file: {args.pdf}", file=sys.stderr)
        return 1
    if min(args.min_text_chars, args.target_chars, args.max_chars, args.top_k, args.candidate_k, args.max_context_chars, args.max_chunk_chars, args.max_tokens, args.timeout_seconds) <= 0:
        print("Numeric limits must be positive.", file=sys.stderr)
        return 1
    if args.target_chars > args.max_chars:
        print("target-chars cannot be larger than max-chars.", file=sys.stderr)
        return 1
    if args.top_k > args.candidate_k:
        print("top-k cannot be larger than candidate-k.", file=sys.stderr)
        return 1

    try:
        result = run_pipeline(
            pdf_path=args.pdf,
            query=query,
            parsed_dir=args.parsed_dir,
            chunks_dir=args.chunks_dir,
            min_text_chars=args.min_text_chars,
            target_chars=args.target_chars,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            embedding_model=args.embedding_model,
            rerank_model=args.rerank_model,
            max_context_chars=args.max_context_chars,
            max_chunk_chars=args.max_chunk_chars,
            local_files_only=not args.allow_download,
            api_key=api_key,
            base_url=args.base_url,
            llm_model=args.llm_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            thinking=args.thinking,
            reasoning_effort=args.reasoning_effort,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        print(f"Failed to run PDF RAG pipeline: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Parsed: {result['parsed_json']}")
        print(f"Chunks: {result['chunks_json']}")
        print(
            "Document: "
            f"pages={result['parsing']['page_count']} "
            f"chunks={result['chunking']['chunk_count']}"
        )
        print()
        print_answer(result["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
