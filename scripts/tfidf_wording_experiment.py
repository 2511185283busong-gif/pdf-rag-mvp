#!/usr/bin/env python3
"""Show how TF-IDF rankings change with wording and a keyword-heavy distractor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from vector_search_demo import (
    build_vocabulary,
    calculate_idf,
    cosine_similarity,
    embed_text,
)
from search_chunks import tokenize


DEFAULT_QUERIES = [
    "How does retrieval work in RAG?",
    "How does RAG find relevant information?",
    "How are useful passages selected for a question?",
    "What powers semantic search?",
    "RAG 如何找到与问题相关的内容？",
]

FAKE_CHUNK = {
    "chunk_id": "fake_keyword_chunk",
    "chunk_index": "fake",
    "text": "The retrieval system retrieves retrieval results.",
    "page_range": "experiment",
    "source_pages": [],
}


def rank_chunks(
    chunks: list[dict[str, Any]], query: str
) -> tuple[list[dict[str, Any]], list[str]]:
    texts = [chunk.get("text", "") for chunk in chunks]
    token_lists = [tokenize(text) for text in texts]
    vocabulary = build_vocabulary(token_lists)
    idf = calculate_idf(token_lists, vocabulary)
    query_vector = embed_text(query, vocabulary, idf)
    active_dimensions = sorted(
        token for token, value in zip(vocabulary, query_vector) if value > 0
    )

    ranked: list[dict[str, Any]] = []
    for chunk, text in zip(chunks, texts):
        vector = embed_text(text, vocabulary, idf)
        ranked.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "score": cosine_similarity(query_vector, vector),
                "pages": chunk.get("page_range"),
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked, active_dimensions


def print_experiment(
    title: str,
    chunks: list[dict[str, Any]],
    queries: list[str],
) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    for query_number, query in enumerate(queries, start=1):
        ranked, active_dimensions = rank_chunks(chunks, query)
        print(f"\nQ{query_number}: {query}")
        dimensions = ", ".join(active_dimensions) or "(none)"
        print(f"  active query dimensions: {dimensions}")
        for rank, item in enumerate(ranked, start=1):
            print(
                f"  {rank}. {item['chunk_id']:<24} "
                f"cosine={item['score']:.6f} pages={item['pages']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare TF-IDF scores across paraphrases and a fake chunk."
    )
    parser.add_argument("chunks_json", type=Path, help="Path to chunks JSON.")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Custom query; repeat this option to test multiple wordings.",
    )
    args = parser.parse_args()

    try:
        data = json.loads(args.chunks_json.read_text(encoding="utf-8"))
        chunks = data.get("chunks", [])
    except Exception as exc:
        print(f"Failed to read chunks: {exc}", file=sys.stderr)
        return 1

    if not chunks:
        print("No chunks found.", file=sys.stderr)
        return 1

    queries = args.queries or DEFAULT_QUERIES
    print("Method: TF-IDF vectors + cosine similarity")
    print(f"Original chunks: {len(chunks)}")
    print(f"Fake chunk: {FAKE_CHUNK['text']}")
    print_experiment("Experiment 1: original chunks", chunks, queries)
    print_experiment(
        "Experiment 2: original chunks + fake keyword chunk",
        [*chunks, FAKE_CHUNK],
        queries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
