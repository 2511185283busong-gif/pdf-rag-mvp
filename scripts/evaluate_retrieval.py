#!/usr/bin/env python3
"""Evaluate embedding retrieval and reranking against labelled PDF questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rerank_search import (
    DEFAULT_RERANK_MODEL,
    rerank_candidates,
    retrieve_candidates,
)
from semantic_search import DEFAULT_MODEL


def compact_result(item: dict[str, Any], rank: int, score_name: str, score: float) -> dict[str, Any]:
    chunk = item["chunk"]
    return {
        "rank": rank,
        "chunk_id": chunk.get("chunk_id"),
        score_name: round(score, 6),
        "page_range": chunk.get("page_range"),
    }


def reciprocal_rank(results: list[dict[str, Any]], relevant_ids: set[str]) -> float:
    for result in results:
        if result["chunk_id"] in relevant_ids:
            return 1.0 / result["rank"]
    return 0.0


def evaluate_retrieval(
    chunks_json_path: Path,
    cases_path: Path,
    top_k: int,
    candidate_k: int,
    embedding_model: str,
    rerank_model: str,
    local_files_only: bool,
) -> dict[str, Any]:
    chunks_data = json.loads(chunks_json_path.read_text(encoding="utf-8"))
    cases_data = json.loads(cases_path.read_text(encoding="utf-8"))
    chunks = chunks_data.get("chunks", [])
    cases = cases_data.get("cases", [])
    known_ids = {chunk.get("chunk_id") for chunk in chunks}

    if not chunks:
        raise ValueError("No chunks found in the input JSON.")
    if not cases:
        raise ValueError("No cases found in the evaluation JSON.")
    if candidate_k < top_k:
        raise ValueError("candidate-k cannot be smaller than top-k.")

    evaluated_cases: list[dict[str, Any]] = []
    embedding_metadata: dict[str, Any] | None = None

    for raw_case in cases:
        case_id = str(raw_case.get("id", "")).strip()
        query = str(raw_case.get("query", "")).strip()
        relevant_ids = {str(value) for value in raw_case.get("relevant_chunk_ids", [])}
        if not case_id or not query or not relevant_ids:
            raise ValueError("Each case needs id, query, and relevant_chunk_ids.")
        unknown_ids = sorted(relevant_ids - known_ids)
        if unknown_ids:
            raise ValueError(f"Case {case_id} references unknown chunk ids: {unknown_ids}")

        candidates, metadata = retrieve_candidates(
            chunks=chunks,
            query=query,
            model_name=embedding_model,
            candidate_k=min(candidate_k, len(chunks)),
            local_files_only=local_files_only,
        )
        embedding_metadata = metadata
        embedding_results = [
            compact_result(item, rank, "cosine", item["base_similarity"])
            for rank, item in enumerate(candidates[:top_k], start=1)
        ]

        reranked = rerank_candidates(
            candidates=candidates,
            query=query,
            model_name=rerank_model,
            local_files_only=local_files_only,
        )
        rerank_results = [
            compact_result(item, rank, "rerank_score", item["rerank_score"])
            for rank, item in enumerate(reranked[:top_k], start=1)
        ]

        evaluated_cases.append(
            {
                "id": case_id,
                "query": query,
                "relevant_chunk_ids": sorted(relevant_ids),
                "verified_pages": raw_case.get("verified_pages", []),
                "embedding": {
                    "results": embedding_results,
                    "hit": any(item["chunk_id"] in relevant_ids for item in embedding_results),
                    "reciprocal_rank": reciprocal_rank(embedding_results, relevant_ids),
                },
                "rerank": {
                    "results": rerank_results,
                    "hit": any(item["chunk_id"] in relevant_ids for item in rerank_results),
                    "reciprocal_rank": reciprocal_rank(rerank_results, relevant_ids),
                },
            }
        )

    case_count = len(evaluated_cases)

    def metrics(stage: str) -> dict[str, float]:
        return {
            f"hit_rate_at_{top_k}": round(
                sum(case[stage]["hit"] for case in evaluated_cases) / case_count, 4
            ),
            f"mrr_at_{top_k}": round(
                sum(case[stage]["reciprocal_rank"] for case in evaluated_cases)
                / case_count,
                4,
            ),
        }

    return {
        "input": {
            "chunks_json": str(chunks_json_path.resolve()),
            "cases_json": str(cases_path.resolve()),
            "chunk_count": len(chunks),
            "case_count": case_count,
        },
        "method": {
            "embedding_model": embedding_metadata["model"] if embedding_metadata else embedding_model,
            "rerank_model": rerank_model,
            "candidate_k": min(candidate_k, len(chunks)),
            "top_k": top_k,
            "uses_llm": False,
        },
        "metrics": {"embedding": metrics("embedding"), "rerank": metrics("rerank")},
        "cases": evaluated_cases,
    }


def print_report(report: dict[str, Any]) -> None:
    method = report["method"]
    print(f"Cases: {report['input']['case_count']} | chunks: {report['input']['chunk_count']}")
    print(f"Embedding model: {method['embedding_model']}")
    print(f"Rerank model: {method['rerank_model']}")
    print(f"Evaluation: candidate_k={method['candidate_k']} top_k={method['top_k']}")
    print()
    print("Embedding metrics:")
    for name, value in report["metrics"]["embedding"].items():
        print(f"- {name}: {value:.4f}")
    print("Rerank metrics:")
    for name, value in report["metrics"]["rerank"].items():
        print(f"- {name}: {value:.4f}")

    for case in report["cases"]:
        print()
        print(f"{case['id']}: {case['query']}")
        print(
            "  embedding: "
            + ", ".join(item["chunk_id"] for item in case["embedding"]["results"])
        )
        print(
            "  rerank:    "
            + ", ".join(item["chunk_id"] for item in case["rerank"]["results"])
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare embedding-only retrieval with embedding plus reranking."
    )
    parser.add_argument("chunks_json", type=Path, help="Path to chunks JSON.")
    parser.add_argument("cases_json", type=Path, help="Labelled evaluation cases JSON.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional path to write the JSON report.")
    args = parser.parse_args()

    if args.top_k <= 0 or args.candidate_k <= 0:
        print("top-k and candidate-k must be positive.", file=sys.stderr)
        return 1

    try:
        report = evaluate_retrieval(
            chunks_json_path=args.chunks_json,
            cases_path=args.cases_json,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            embedding_model=args.embedding_model,
            rerank_model=args.rerank_model,
            local_files_only=not args.allow_download,
        )
    except Exception as exc:
        print(f"Failed to evaluate retrieval: {exc}", file=sys.stderr)
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
