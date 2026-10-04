#!/usr/bin/env python3
"""Build an LLM-ready context block from reranked PDF chunks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from rerank_search import DEFAULT_RERANK_MODEL, rerank_search
from semantic_search import DEFAULT_MODEL


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def trim_text(text: str, max_chars: int) -> tuple[str, bool]:
    compact = compact_text(text)
    if len(compact) <= max_chars:
        return compact, False
    if max_chars <= 0:
        return "", True
    ellipsis = "..."
    if max_chars <= len(ellipsis):
        return ellipsis[:max_chars], True
    prefix = compact[: max_chars - len(ellipsis)].rstrip()
    return f"{prefix}{ellipsis}", True


def format_context_header(source: dict[str, Any]) -> str:
    return (
        f"[{source['source_id']}] "
        f"chunk={source['chunk_id']} "
        f"pages={source['page_range']} "
        f"rerank={source['rerank_score']:.6f} "
        f"base_rank={source['base_rank']}"
    )


def format_context_source(source: dict[str, Any]) -> str:
    return f"{format_context_header(source)}\n{source['text']}"


def build_context(
    chunks_json_path: Path,
    query: str,
    top_k: int,
    candidate_k: int,
    embedding_model: str,
    rerank_model: str,
    max_context_chars: int,
    max_chunk_chars: int,
    local_files_only: bool = True,
) -> dict[str, Any]:
    data = json.loads(chunks_json_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    chunks_by_id = {chunk.get("chunk_id"): chunk for chunk in chunks}

    reranked = rerank_search(
        chunks_json_path=chunks_json_path,
        query=query,
        top_k=top_k,
        candidate_k=candidate_k,
        embedding_model=embedding_model,
        rerank_model=rerank_model,
        snippet_chars=max_chunk_chars,
        local_files_only=local_files_only,
    )

    sources: list[dict[str, Any]] = []
    used_chars = 0
    for result in reranked["results"]:
        chunk = chunks_by_id.get(result["chunk_id"])
        if not chunk:
            continue

        source = {
            "source_id": f"S{len(sources) + 1}",
            "chunk_id": result["chunk_id"],
            "chunk_index": result["chunk_index"],
            "page_range": result["page_range"],
            "source_pages": result["source_pages"],
            "rerank_rank": result["rerank_rank"],
            "rerank_score": result["rerank_score"],
            "base_rank": result["base_rank"],
            "base_similarity": result["base_similarity"],
        }

        separator_chars = 2 if sources else 0
        fixed_chars = separator_chars + len(format_context_header(source)) + 1
        available_text_chars = max_context_chars - used_chars - fixed_chars
        if available_text_chars <= 0:
            break

        allowed_chunk_chars = min(max_chunk_chars, available_text_chars)
        text, truncated = trim_text(chunk.get("text", ""), allowed_chunk_chars)
        if not text:
            continue

        source.update(
            {
                "text": text,
                "included_chars": len(text),
                "truncated": truncated,
            }
        )
        sources.append(source)
        used_chars += fixed_chars + len(text)

    context = "\n\n".join(format_context_source(source) for source in sources)
    prompt = "\n\n".join(
        [
            "Answer the question using only the context below.",
            "If the context is not enough, say that the document does not contain enough information.",
            "Cite sources with source ids and pages, for example [S1, pages 1-2].",
            f"Question: {query}",
            f"Context:\n{context}",
        ]
    )

    return {
        "query": query,
        "context": context,
        "prompt": prompt,
        "sources": sources,
        "budget": {
            "max_context_chars": max_context_chars,
            "max_chunk_chars": max_chunk_chars,
            "used_context_chars": len(context),
        },
        "method": reranked["method"],
        "source": data.get("source", {}),
        "input": reranked["input"],
    }


def print_context(result: dict[str, Any], include_prompt: bool) -> None:
    print(f"Question: {result['query']}")
    print(
        "Context budget: "
        f"{result['budget']['used_context_chars']}/"
        f"{result['budget']['max_context_chars']} chars"
    )
    print()

    if include_prompt:
        print(result["prompt"])
        return

    print("Context:")
    print(result["context"])
    print()
    print("Sources:")
    for source in result["sources"]:
        suffix = " truncated" if source["truncated"] else ""
        print(
            f"- [{source['source_id']}] {source['chunk_id']} "
            f"pages={source['page_range']} "
            f"rerank={source['rerank_score']:.6f}{suffix}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a citation-friendly context block from reranked chunks."
    )
    parser.add_argument("chunks_json", type=Path, help="Path to chunks JSON.")
    parser.add_argument("query", nargs="+", help="Question to build context for.")
    parser.add_argument("--top-k", type=int, default=3, help="Sources to include.")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Number of embedding candidates passed into rerank.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=3000,
        help="Maximum total context characters.",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=1200,
        help="Maximum characters copied from each chunk.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_MODEL,
        help=f"Sentence Transformers embedding model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--rerank-model",
        default=DEFAULT_RERANK_MODEL,
        help=f"CrossEncoder rerank model (default: {DEFAULT_RERANK_MODEL}).",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Print a full prompt instead of only the context block.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text output.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow model downloads if a model is missing from local cache.",
    )
    args = parser.parse_args()
    query = " ".join(args.query).strip()

    if not query:
        print("query cannot be empty.", file=sys.stderr)
        return 1
    if args.top_k <= 0 or args.candidate_k <= 0:
        print("top-k and candidate-k must be positive.", file=sys.stderr)
        return 1
    if args.top_k > args.candidate_k:
        print("top-k cannot be larger than candidate-k.", file=sys.stderr)
        return 1
    if args.max_context_chars <= 0 or args.max_chunk_chars <= 0:
        print("max-context-chars and max-chunk-chars must be positive.", file=sys.stderr)
        return 1

    try:
        result = build_context(
            chunks_json_path=args.chunks_json,
            query=query,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            embedding_model=args.embedding_model,
            rerank_model=args.rerank_model,
            max_context_chars=args.max_context_chars,
            max_chunk_chars=args.max_chunk_chars,
            local_files_only=not args.allow_download,
        )
    except Exception as exc:
        print(f"Failed to build context: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_context(result, args.prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
