#!/usr/bin/env python3
"""Compare dense embedding rankings across paraphrases and a fake chunk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from semantic_search import DEFAULT_MODEL, MODEL_CACHE, resolve_model_reference
from tfidf_wording_experiment import DEFAULT_QUERIES, FAKE_CHUNK


def rank_chunks(
    chunks: list[dict[str, Any]],
    query_vector: Any,
    passage_vectors: Any,
) -> list[dict[str, Any]]:
    similarities = passage_vectors @ query_vector
    ranked = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "score": float(score),
            "pages": chunk.get("page_range"),
        }
        for chunk, score in zip(chunks, similarities)
    ]
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def print_rankings(
    title: str,
    chunks: list[dict[str, Any]],
    queries: list[str],
    query_vectors: Any,
    passage_vectors: Any,
) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    for query_number, (query, query_vector) in enumerate(
        zip(queries, query_vectors), start=1
    ):
        print(f"\nQ{query_number}: {query}")
        for rank, item in enumerate(
            rank_chunks(chunks, query_vector, passage_vectors), start=1
        ):
            print(
                f"  {rank}. {item['chunk_id']:<24} "
                f"cosine={item['score']:.6f} pages={item['pages']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare dense embedding scores across paraphrases and a fake chunk."
    )
    parser.add_argument("chunks_json", type=Path, help="Path to chunks JSON.")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Custom query; repeat this option to test multiple wordings.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Sentence Transformers model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow a model download if it is missing from local cache.",
    )
    args = parser.parse_args()

    local_files_only = not args.allow_download

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "sentence-transformers is not installed. Run: "
            "python3 -m pip install -r requirements-semantic.txt",
            file=sys.stderr,
        )
        return 1

    try:
        data = json.loads(args.chunks_json.read_text(encoding="utf-8"))
        original_chunks = data.get("chunks", [])
    except Exception as exc:
        print(f"Failed to read chunks: {exc}", file=sys.stderr)
        return 1

    if not original_chunks:
        print("No chunks found.", file=sys.stderr)
        return 1

    queries = args.queries or DEFAULT_QUERIES
    all_chunks = [*original_chunks, FAKE_CHUNK]
    try:
        model_reference = resolve_model_reference(args.model, local_files_only)
        model = SentenceTransformer(
            model_reference,
            cache_folder=str(MODEL_CACHE),
            local_files_only=local_files_only,
        )
    except Exception as exc:
        if local_files_only:
            print(
                f"Embedding model is not available in local cache: {args.model}. "
                "Run again with --allow-download once while online.",
                file=sys.stderr,
            )
            return 1
        raise
    query_vectors = model.encode(
        [f"query: {query}" for query in queries],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    passage_vectors = model.encode(
        [f"passage: {chunk.get('text', '')}" for chunk in all_chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    print("Method: dense neural embeddings + cosine similarity")
    print(f"Model: {args.model}")
    print(f"Vector dimensions: {query_vectors.shape[1]}")
    print(f"Original chunks: {len(original_chunks)}")
    print(f"Fake chunk: {FAKE_CHUNK['text']}")
    print_rankings(
        "Experiment 1: original chunks",
        original_chunks,
        queries,
        query_vectors,
        passage_vectors[: len(original_chunks)],
    )
    print_rankings(
        "Experiment 2: original chunks + fake keyword chunk",
        all_chunks,
        queries,
        query_vectors,
        passage_vectors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
