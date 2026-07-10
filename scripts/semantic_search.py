#!/usr/bin/env python3
"""Dense semantic search over local chunks with page citations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from search_chunks import compact_text


DEFAULT_MODEL = "intfloat/multilingual-e5-small"
MODEL_CACHE = Path(__file__).resolve().parents[1] / ".models"
os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(MODEL_CACHE / "xet"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def configure_model_loading(local_files_only: bool) -> None:
    """Avoid Hub metadata calls when a cached model is intentionally offline."""
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def make_snippet(text: str, max_chars: int) -> str:
    compact = compact_text(text)
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].rstrip()}..."


def semantic_search(
    chunks_json_path: Path,
    query: str,
    top_k: int,
    model_name: str,
    snippet_chars: int,
) -> dict[str, Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Run: "
            "python3 -m pip install -r requirements-semantic.txt"
        ) from exc

    data = json.loads(chunks_json_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    if not chunks:
        raise ValueError("No chunks found in the input JSON.")

    model = SentenceTransformer(model_name, cache_folder=str(MODEL_CACHE))
    passage_texts = [f"passage: {chunk.get('text', '')}" for chunk in chunks]
    query_text = f"query: {query}"

    # Normalized embeddings turn a dot product into cosine similarity.
    passage_vectors = model.encode(
        passage_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_vector = model.encode(
        query_text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    similarities = passage_vectors @ query_vector

    results: list[dict[str, Any]] = []
    for chunk, similarity in zip(chunks, similarities):
        results.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": chunk.get("chunk_index"),
                "similarity": round(float(similarity), 6),
                "page_range": chunk.get("page_range"),
                "source_pages": chunk.get("source_pages", []),
                "snippet": make_snippet(chunk.get("text", ""), snippet_chars),
            }
        )

    results.sort(key=lambda item: item["similarity"], reverse=True)
    dimension = int(query_vector.shape[-1])

    return {
        "query": query,
        "method": {
            "embedding": "dense_neural_embedding",
            "model": model_name,
            "vector_dimension": dimension,
            "similarity": "cosine",
            "query_prefix": "query:",
            "passage_prefix": "passage:",
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
    print(f"Model: {method['model']}")
    print(
        f"Embedding: dense neural vector | dimensions={method['vector_dimension']}"
    )
    print("Similarity: cosine")
    print("Vector database: no")
    print("LLM: no")

    for index, item in enumerate(result["results"], start=1):
        print()
        print(
            f"{index}. {item['chunk_id']} | cosine={item['similarity']:.6f} | "
            f"pages={item['page_range']}"
        )
        print(item["snippet"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank chunks with a real multilingual neural embedding model."
    )
    parser.add_argument("chunks_json", type=Path, help="Path to chunks JSON.")
    parser.add_argument("query", nargs="+", help="Question to search for.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Sentence Transformers model (default: {DEFAULT_MODEL}).",
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
    if args.top_k <= 0:
        print("top-k must be positive.", file=sys.stderr)
        return 1

    try:
        result = semantic_search(
            args.chunks_json,
            query,
            args.top_k,
            args.model,
            args.snippet_chars,
        )
    except Exception as exc:
        print(f"Failed to run semantic search: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_results(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
