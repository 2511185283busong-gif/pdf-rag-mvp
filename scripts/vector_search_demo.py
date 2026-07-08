#!/usr/bin/env python3
"""Local TF-IDF vector search demo with cosine similarity and page citations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from search_chunks import compact_text, tokenize


def build_vocabulary(chunk_token_lists: list[list[str]]) -> list[str]:
    return sorted({token for tokens in chunk_token_lists for token in tokens})


def calculate_idf(
    chunk_token_lists: list[list[str]], vocabulary: list[str]
) -> dict[str, float]:
    document_count = len(chunk_token_lists)
    document_frequency = Counter(
        token for tokens in chunk_token_lists for token in set(tokens)
    )
    return {
        token: math.log((document_count + 1) / (document_frequency[token] + 1)) + 1
        for token in vocabulary
    }


def embed_text(
    text: str,
    vocabulary: list[str],
    idf: dict[str, float],
) -> list[float]:
    tokens = tokenize(text)
    frequencies = Counter(tokens)
    token_count = max(len(tokens), 1)
    return [
        (frequencies[token] / token_count) * idf[token]
        for token in vocabulary
    ]


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def top_shared_dimensions(
    query_vector: list[float],
    chunk_vector: list[float],
    vocabulary: list[str],
    limit: int = 8,
) -> list[dict[str, Any]]:
    contributions = [
        (token, query_value * chunk_value)
        for token, query_value, chunk_value in zip(
            vocabulary, query_vector, chunk_vector
        )
        if query_value > 0 and chunk_value > 0
    ]
    contributions.sort(key=lambda item: item[1], reverse=True)
    return [
        {"token": token, "contribution": round(value, 8)}
        for token, value in contributions[:limit]
    ]


def top_vector_dimensions(
    vector: list[float],
    vocabulary: list[str],
    limit: int = 12,
) -> list[dict[str, Any]]:
    dimensions = [
        (token, value)
        for token, value in zip(vocabulary, vector)
        if value > 0
    ]
    dimensions.sort(key=lambda item: item[1], reverse=True)
    return [
        {"token": token, "value": round(value, 8)}
        for token, value in dimensions[:limit]
    ]


def make_snippet(text: str, max_chars: int) -> str:
    compact = compact_text(text)
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].rstrip()}..."


def vector_search(
    chunks_json_path: Path,
    query: str,
    top_k: int,
    chunk_limit: int,
    snippet_chars: int,
) -> dict[str, Any]:
    data = json.loads(chunks_json_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])[:chunk_limit]
    chunk_token_lists = [tokenize(chunk.get("text", "")) for chunk in chunks]
    vocabulary = build_vocabulary(chunk_token_lists)
    idf = calculate_idf(chunk_token_lists, vocabulary)

    chunk_vectors = [
        embed_text(chunk.get("text", ""), vocabulary, idf) for chunk in chunks
    ]
    query_vector = embed_text(query, vocabulary, idf)
    non_zero_query_dimensions = sum(1 for value in query_vector if value > 0)

    results: list[dict[str, Any]] = []
    for chunk, chunk_vector in zip(chunks, chunk_vectors):
        similarity = cosine_similarity(query_vector, chunk_vector)
        results.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": chunk.get("chunk_index"),
                "similarity": round(similarity, 6),
                "page_range": chunk.get("page_range"),
                "source_pages": chunk.get("source_pages", []),
                "shared_dimensions": top_shared_dimensions(
                    query_vector, chunk_vector, vocabulary
                ),
                "snippet": make_snippet(chunk.get("text", ""), snippet_chars),
            }
        )

    results.sort(key=lambda result: result["similarity"], reverse=True)

    return {
        "query": query,
        "method": {
            "embedding": "local_tfidf_demo",
            "similarity": "cosine",
            "vector_dimension": len(vocabulary),
            "query_non_zero_dimensions": non_zero_query_dimensions,
            "query_top_dimensions": top_vector_dimensions(
                query_vector, vocabulary
            ),
            "uses_vector_database": False,
            "uses_llm": False,
        },
        "source": data.get("source", {}),
        "input": {
            "path": str(chunks_json_path.resolve()),
            "chunk_count": len(chunks),
        },
        "results": results[:top_k],
    }


def print_results(result: dict[str, Any]) -> None:
    method = result["method"]
    print(f"Query: {result['query']}")
    print(
        "Embedding: "
        f"{method['embedding']} | dimensions={method['vector_dimension']} | "
        f"query_non_zero={method['query_non_zero_dimensions']}"
    )
    print("Similarity: cosine")
    print("Vector database: no")
    print("LLM: no")
    query_dimensions = ", ".join(
        f"{item['token']}={item['value']:.4f}"
        for item in method["query_top_dimensions"]
    )
    print(f"Query vector dimensions: {query_dimensions or '(none)'}")

    if method["query_non_zero_dimensions"] == 0:
        print()
        print("The query has no dimensions in the local vocabulary.")
        print("Try words that appear in the document.")
        return

    for index, item in enumerate(result["results"], start=1):
        shared = ", ".join(
            f"{dimension['token']}={dimension['contribution']:.6f}"
            for dimension in item["shared_dimensions"]
        )
        print()
        print(
            f"{index}. {item['chunk_id']} | cosine={item['similarity']:.6f} | "
            f"pages={item['page_range']}"
        )
        print(f"shared_dimensions: {shared or '(none)'}")
        print(item["snippet"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Turn chunks and a query into local TF-IDF vectors, then rank by cosine similarity."
    )
    parser.add_argument(
        "chunks_json",
        type=Path,
        help="Path to chunks JSON, for example chunks/resume2_chunks.json.",
    )
    parser.add_argument("query", nargs="+", help="Question or text to embed.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results.")
    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=10,
        help="Maximum number of chunks used in this teaching demo.",
    )
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

    if not query:
        print("query cannot be empty.", file=sys.stderr)
        return 1
    if args.top_k <= 0 or args.chunk_limit <= 0:
        print("top-k and chunk-limit must be positive.", file=sys.stderr)
        return 1

    try:
        result = vector_search(
            args.chunks_json,
            query,
            args.top_k,
            args.chunk_limit,
            args.snippet_chars,
        )
    except Exception as exc:
        print(f"Failed to run vector search demo: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_results(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
