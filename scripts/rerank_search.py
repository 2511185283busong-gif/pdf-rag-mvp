#!/usr/bin/env python3
"""Two-stage retrieval: dense embedding recall followed by cross-encoder rerank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from semantic_search import DEFAULT_MODEL, MODEL_CACHE, make_snippet


DEFAULT_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def retrieve_candidates(
    chunks: list[dict[str, Any]],
    query: str,
    model_name: str,
    candidate_k: int,
    local_files_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(
            model_name,
            cache_folder=str(MODEL_CACHE),
            local_files_only=local_files_only,
        )
    except Exception as exc:
        if local_files_only:
            raise RuntimeError(
                f"Embedding model is not available in local cache: {model_name}. "
                "Run again with --allow-download once while online."
            ) from exc
        raise
    passage_texts = [f"passage: {chunk.get('text', '')}" for chunk in chunks]
    query_text = f"query: {query}"

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

    candidates: list[dict[str, Any]] = []
    for base_rank, (chunk, similarity) in enumerate(
        sorted(
            zip(chunks, similarities),
            key=lambda item: float(item[1]),
            reverse=True,
        ),
        start=1,
    ):
        candidates.append(
            {
                "chunk": chunk,
                "base_rank": base_rank,
                "base_similarity": float(similarity),
            }
        )

    metadata = {
        "model": model_name,
        "vector_dimension": int(query_vector.shape[-1]),
        "similarity": "cosine",
        "query_prefix": "query:",
        "passage_prefix": "passage:",
    }
    return candidates[:candidate_k], metadata


def rerank_candidates(
    candidates: list[dict[str, Any]],
    query: str,
    model_name: str,
    local_files_only: bool,
) -> list[dict[str, Any]]:
    from sentence_transformers import CrossEncoder

    try:
        model = CrossEncoder(
            model_name,
            cache_folder=str(MODEL_CACHE),
            local_files_only=local_files_only,
        )
    except Exception as exc:
        if local_files_only:
            raise RuntimeError(
                f"Rerank model is not available in local cache: {model_name}. "
                "Run again with --allow-download once while online."
            ) from exc
        raise
    pairs = [(query, item["chunk"].get("text", "")) for item in candidates]
    scores = model.predict(pairs, show_progress_bar=False)

    reranked: list[dict[str, Any]] = []
    for candidate, score in zip(candidates, scores):
        reranked.append(
            {
                **candidate,
                "rerank_score": float(score),
            }
        )
    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked


def rerank_search(
    chunks_json_path: Path,
    query: str,
    top_k: int,
    candidate_k: int,
    embedding_model: str,
    rerank_model: str,
    snippet_chars: int,
    local_files_only: bool = True,
) -> dict[str, Any]:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Run: "
            "python3 -m pip install -r requirements-semantic.txt"
        ) from exc

    data = json.loads(chunks_json_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    if not chunks:
        raise ValueError("No chunks found in the input JSON.")

    candidate_k = min(candidate_k, len(chunks))
    candidates, retrieval_metadata = retrieve_candidates(
        chunks=chunks,
        query=query,
        model_name=embedding_model,
        candidate_k=candidate_k,
        local_files_only=local_files_only,
    )
    reranked = rerank_candidates(
        candidates=candidates,
        query=query,
        model_name=rerank_model,
        local_files_only=local_files_only,
    )

    results: list[dict[str, Any]] = []
    for rerank_rank, item in enumerate(reranked[:top_k], start=1):
        chunk = item["chunk"]
        results.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": chunk.get("chunk_index"),
                "rerank_rank": rerank_rank,
                "rerank_score": round(item["rerank_score"], 6),
                "base_rank": item["base_rank"],
                "base_similarity": round(item["base_similarity"], 6),
                "page_range": chunk.get("page_range"),
                "source_pages": chunk.get("source_pages", []),
                "snippet": make_snippet(chunk.get("text", ""), snippet_chars),
            }
        )

    return {
        "query": query,
        "method": {
            "pipeline": "dense_embedding_recall_then_cross_encoder_rerank",
            "retrieval": retrieval_metadata,
            "rerank": {
                "model": rerank_model,
                "input": "query + candidate chunk text",
                "score": "cross_encoder_relevance_logit",
            },
            "candidate_k": candidate_k,
            "top_k": top_k,
            "uses_vector_database": False,
            "uses_llm": False,
        },
        "source": data.get("source", {}),
        "input": {
            "path": str(chunks_json_path.resolve()),
            "chunk_count": len(chunks),
        },
        "results": results,
    }


def print_results(result: dict[str, Any]) -> None:
    method = result["method"]
    retrieval = method["retrieval"]
    rerank = method["rerank"]

    print(f"Query: {result['query']}")
    print(f"Recall model: {retrieval['model']}")
    print(f"Recall: dense embedding cosine | candidates={method['candidate_k']}")
    print(f"Rerank model: {rerank['model']}")
    print("Rerank: cross-encoder query+chunk relevance")
    print("Vector database: no")
    print("LLM: no")

    for index, item in enumerate(result["results"], start=1):
        print()
        print(
            f"{index}. {item['chunk_id']} | rerank={item['rerank_score']:.6f} | "
            f"base_rank={item['base_rank']} | base_cosine={item['base_similarity']:.6f} | "
            f"pages={item['page_range']}"
        )
        print(item["snippet"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve chunks with dense embeddings, then rerank candidates with a cross-encoder."
    )
    parser.add_argument("chunks_json", type=Path, help="Path to chunks JSON.")
    parser.add_argument("query", nargs="+", help="Question to search for.")
    parser.add_argument("--top-k", type=int, default=3, help="Final results to show.")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Number of embedding candidates to pass into the reranker.",
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

    try:
        result = rerank_search(
            chunks_json_path=args.chunks_json,
            query=query,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            embedding_model=args.embedding_model,
            rerank_model=args.rerank_model,
            snippet_chars=args.snippet_chars,
            local_files_only=not args.allow_download,
        )
    except Exception as exc:
        print(f"Failed to run rerank search: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_results(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
