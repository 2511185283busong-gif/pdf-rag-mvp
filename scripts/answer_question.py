#!/usr/bin/env python3
"""Answer a PDF question with DeepSeek using reranked context."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from context_builder import build_context
from rerank_search import DEFAULT_RERANK_MODEL
from semantic_search import DEFAULT_MODEL


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_messages(query: str, context: str) -> list[dict[str, str]]:
    system_prompt = (
        "You are a careful RAG assistant. Answer using only the provided context. "
        "If the context does not contain enough information, say so clearly. "
        "Do not invent facts. Answer in the same language as the user's question. "
        "Cite source ids and pages, for example [S1, pages 1-2]."
    )
    user_prompt = "\n\n".join(
        [
            f"Question: {query}",
            f"Context:\n{context}",
            "Write a concise answer with citations.",
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_deepseek(
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    thinking: str,
    reasoning_effort: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "thinking": {"type": thinking},
    }

    if thinking == "enabled":
        payload["reasoning_effort"] = reasoning_effort
    else:
        payload["temperature"] = temperature

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API returned HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to connect to DeepSeek API: {exc.reason}") from exc


def extract_answer(response: dict[str, Any]) -> str:
    try:
        return response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise RuntimeError(f"Unexpected DeepSeek response shape: {response}") from exc


def answer_question(
    chunks_json_path: Path,
    query: str,
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
    context_result = build_context(
        chunks_json_path=chunks_json_path,
        query=query,
        top_k=top_k,
        candidate_k=candidate_k,
        embedding_model=embedding_model,
        rerank_model=rerank_model,
        max_context_chars=max_context_chars,
        max_chunk_chars=max_chunk_chars,
        local_files_only=local_files_only,
    )
    messages = build_messages(query, context_result["context"])
    response = call_deepseek(
        api_key=api_key,
        base_url=base_url,
        model=llm_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout_seconds,
    )

    return {
        "query": query,
        "answer": extract_answer(response),
        "sources": context_result["sources"],
        "context_budget": context_result["budget"],
        "retrieval_method": context_result["method"],
        "llm": {
            "provider": "deepseek",
            "model": llm_model,
            "base_url": base_url,
            "thinking": thinking,
            "reasoning_effort": reasoning_effort if thinking == "enabled" else None,
        },
        "usage": response.get("usage", {}),
    }


def print_answer(result: dict[str, Any]) -> None:
    print(f"Question: {result['query']}")
    print()
    print(result["answer"])
    print()
    print("Sources:")
    for source in result["sources"]:
        print(
            f"- [{source['source_id']}] {source['chunk_id']} "
            f"pages={source['page_range']} "
            f"rerank={source['rerank_score']:.6f}"
        )
    if result["usage"]:
        print()
        print(f"Token usage: {result['usage']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Answer a question over PDF chunks using DeepSeek and reranked context."
    )
    parser.add_argument("chunks_json", type=Path, help="Path to chunks JSON.")
    parser.add_argument("query", nargs="+", help="Question to answer.")
    parser.add_argument("--top-k", type=int, default=2, help="Sources to include.")
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
        "--allow-download",
        action="store_true",
        help="Allow embedding/rerank model downloads if missing from local cache.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional env file containing DEEPSEEK_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
        help=f"DeepSeek API base URL (default: {DEFAULT_DEEPSEEK_BASE_URL}).",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        help=f"DeepSeek model (default: {DEFAULT_DEEPSEEK_MODEL}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=600,
        help="Maximum answer tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature when thinking is disabled.",
    )
    parser.add_argument(
        "--thinking",
        choices=["disabled", "enabled"],
        default="disabled",
        help="DeepSeek thinking mode.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["high", "max"],
        default="high",
        help="Reasoning effort when thinking is enabled.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="DeepSeek API request timeout.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text output.",
    )
    args = parser.parse_args()
    query = " ".join(args.query).strip()

    load_env_file(args.env_file)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()

    if not query:
        print("query cannot be empty.", file=sys.stderr)
        return 1
    if not api_key:
        print(
            "DEEPSEEK_API_KEY is not set. Put it in .env or run: "
            "export DEEPSEEK_API_KEY='your_api_key'",
            file=sys.stderr,
        )
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
    if args.max_tokens <= 0 or args.timeout_seconds <= 0:
        print("max-tokens and timeout-seconds must be positive.", file=sys.stderr)
        return 1

    try:
        result = answer_question(
            chunks_json_path=args.chunks_json,
            query=query,
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
        print(f"Failed to answer question: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_answer(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
