#!/usr/bin/env python3
"""Keyword search over chunk JSON and return matching chunks with page citations."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    raw_latin_tokens = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", text)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", text)

    latin_tokens: list[str] = []
    for token in raw_latin_tokens:
        latin_tokens.append(token)
        latin_tokens.extend(part for part in re.split(r"[-']", token) if part)
        if token.endswith("ies") and len(token) > 4:
            latin_tokens.append(f"{token[:-3]}y")
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            latin_tokens.append(token[:-1])

    chinese_tokens: list[str] = []
    for run in chinese_runs:
        chinese_tokens.extend(run)
        if len(run) > 1:
            chinese_tokens.extend(run[index : index + 2] for index in range(len(run) - 1))

    tokens = latin_tokens + chinese_tokens
    return [token for token in tokens if token not in STOPWORDS and len(token.strip()) > 0]


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def make_snippet(text: str, query: str, query_tokens: list[str], max_chars: int) -> str:
    compact = compact_text(text)
    lower = compact.lower()
    normalized_query = compact_text(query).lower()

    first_match = lower.find(normalized_query) if len(normalized_query) >= 3 else -1
    if first_match < 0:
        scored_positions: list[tuple[int, int]] = []
        for token in set(query_tokens):
            for match in re.finditer(re.escape(token.lower()), lower):
                start = max(0, match.start() - max_chars // 4)
                end = min(len(lower), start + max_chars)
                local_score = sum(
                    1 for other in set(query_tokens) if other in lower[start:end]
                )
                scored_positions.append((local_score, match.start()))

        if scored_positions:
            first_match = max(scored_positions, key=lambda item: (item[0], -item[1]))[1]

    if first_match < 0:
        return compact[:max_chars]

    start = max(0, first_match - max_chars // 3)
    end = min(len(compact), start + max_chars)
    snippet = compact[start:end].strip()
    if start > 0:
        snippet = f"...{snippet}"
    if end < len(compact):
        snippet = f"{snippet}..."
    return snippet


def bm25_score(
    query_tokens: list[str],
    chunk_tokens: list[str],
    document_frequency: dict[str, int],
    total_documents: int,
    average_length: float,
) -> float:
    if not query_tokens or not chunk_tokens:
        return 0.0

    frequencies = Counter(chunk_tokens)
    chunk_length = len(chunk_tokens)
    k1 = 1.5
    b = 0.75
    score = 0.0

    for token in set(query_tokens):
        frequency = frequencies[token]
        if frequency == 0:
            continue
        df = document_frequency.get(token, 0)
        idf = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))
        denominator = frequency + k1 * (1 - b + b * chunk_length / average_length)
        score += idf * frequency * (k1 + 1) / denominator

    return score


def matched_terms(query_tokens: list[str], chunk_tokens: list[str]) -> list[str]:
    chunk_token_set = set(chunk_tokens)
    return sorted(token for token in set(query_tokens) if token in chunk_token_set)


def phrase_bonus(query: str, text: str) -> float:
    normalized_query = compact_text(query).lower()
    normalized_text = compact_text(text).lower()
    if len(normalized_query) < 3:
        return 0.0
    if normalized_query in normalized_text:
        return 1.5
    return 0.0


def search_chunks(
    chunks_json_path: Path,
    query: str,
    top_k: int,
    snippet_chars: int,
) -> dict[str, Any]:
    data = json.loads(chunks_json_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    query_tokens = tokenize(query)

    chunk_token_lists = [tokenize(chunk.get("text", "")) for chunk in chunks]
    document_frequency: dict[str, int] = {}
    for tokens in chunk_token_lists:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    average_length = sum(len(tokens) for tokens in chunk_token_lists) / max(len(chunks), 1)
    average_length = max(average_length, 1.0)

    results: list[dict[str, Any]] = []
    for chunk, tokens in zip(chunks, chunk_token_lists):
        score = bm25_score(
            query_tokens=query_tokens,
            chunk_tokens=tokens,
            document_frequency=document_frequency,
            total_documents=len(chunks),
            average_length=average_length,
        )
        score += phrase_bonus(query, chunk.get("text", ""))
        if score <= 0:
            continue

        results.append(
            {
                "score": round(score, 4),
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": chunk.get("chunk_index"),
                "page_range": chunk.get("page_range"),
                "source_pages": chunk.get("source_pages", []),
                "matched_terms": matched_terms(query_tokens, tokens),
                "char_count": chunk.get("char_count"),
                "snippet": make_snippet(
                    chunk.get("text", ""), query, query_tokens, snippet_chars
                ),
                "text": chunk.get("text", ""),
            }
        )

    results.sort(key=lambda result: result["score"], reverse=True)

    return {
        "query": query,
        "query_tokens": query_tokens,
        "source": data.get("source", {}),
        "input": {
            "path": str(chunks_json_path.resolve()),
            "chunk_count": len(chunks),
        },
        "results": results[:top_k],
    }


def print_text_results(result: dict[str, Any]) -> None:
    print(f"Query: {result['query']}")
    print(f"Chunks searched: {result['input']['chunk_count']}")
    print(f"Matched chunks: {len(result['results'])}")

    if not result["results"]:
        print("No keyword matches found.")
        return

    for index, item in enumerate(result["results"], start=1):
        print()
        print(
            f"{index}. {item['chunk_id']} | score={item['score']} | "
            f"pages={item['page_range']}"
        )
        print(f"matched_terms: {', '.join(item['matched_terms'])}")
        print(item["snippet"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search chunk JSON with lightweight keyword retrieval."
    )
    parser.add_argument(
        "chunks_json",
        type=Path,
        help="Path to chunks JSON, for example chunks/resume2_chunks.json.",
    )
    parser.add_argument("query", nargs="+", help="Question or keywords to search for.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to show.")
    parser.add_argument(
        "--snippet-chars",
        type=int,
        default=260,
        help="Maximum snippet length per result.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text output.",
    )

    args = parser.parse_args()
    query = " ".join(args.query).strip()

    if args.top_k <= 0:
        print("top-k must be positive.", file=sys.stderr)
        return 1
    if not query:
        print("query cannot be empty.", file=sys.stderr)
        return 1

    try:
        result = search_chunks(args.chunks_json, query, args.top_k, args.snippet_chars)
    except Exception as exc:
        print(f"Failed to search chunks: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_text_results(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
