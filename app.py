"""Small Streamlit showcase for the local PDF RAG pipeline."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from answer_question import (  # noqa: E402
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    load_env_file,
)
from ask_pdf import run_pipeline  # noqa: E402
from chunk_json import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP_CHARS, DEFAULT_TARGET_CHARS  # noqa: E402
from rerank_search import DEFAULT_RERANK_MODEL  # noqa: E402
from semantic_search import DEFAULT_MODEL  # noqa: E402


st.set_page_config(page_title="PDF RAG MVP", page_icon="PDF", layout="wide")
st.markdown(
    """
    <style>
    @media (max-width: 640px) {
        .stMainBlockContainer { padding: 2rem 1rem 3rem; }
        .stApp h1 { font-size: 2.25rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_api_key() -> str:
    load_env_file(ROOT / ".env")
    return os.environ.get("DEEPSEEK_API_KEY", "").strip()


def render_sources(sources: list[dict]) -> None:
    st.subheader("Sources")
    for source in sources:
        st.markdown(
            f"**[{source['source_id']}] {source['chunk_id']}**  \nPages {source['page_range']} · Rerank {source['rerank_score']:.3f}"
        )


def run_question(uploaded_pdf: st.runtime.uploaded_file_manager.UploadedFile, query: str, candidate_k: int, top_k: int, allow_download: bool) -> dict:
    with tempfile.TemporaryDirectory(prefix="pdf-rag-demo-") as directory:
        work_dir = Path(directory)
        pdf_path = work_dir / Path(uploaded_pdf.name).name
        pdf_path.write_bytes(uploaded_pdf.getvalue())

        return run_pipeline(
            pdf_path=pdf_path,
            query=query,
            parsed_dir=work_dir / "parsed",
            chunks_dir=work_dir / "chunks",
            min_text_chars=20,
            target_chars=DEFAULT_TARGET_CHARS,
            max_chars=DEFAULT_MAX_CHARS,
            overlap_chars=DEFAULT_OVERLAP_CHARS,
            top_k=top_k,
            candidate_k=candidate_k,
            embedding_model=DEFAULT_MODEL,
            rerank_model=DEFAULT_RERANK_MODEL,
            max_context_chars=3000,
            max_chunk_chars=1200,
            local_files_only=not allow_download,
            api_key=load_api_key(),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
            llm_model=os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
            max_tokens=600,
            temperature=0.2,
            thinking="disabled",
            reasoning_effort="high",
            timeout_seconds=60,
        )


st.title("PDF RAG MVP")
st.caption("Upload a text-layer PDF, ask a question, and inspect the cited evidence.")

with st.sidebar:
    st.header("Retrieval settings")
    candidate_k = st.number_input("Recall candidates", min_value=1, max_value=50, value=20)
    top_k = st.number_input("Sources for answer", min_value=1, max_value=5, value=2)
    allow_download = st.checkbox("Allow first-time model download", value=False)
    st.divider()
    st.caption("Text-layer PDFs only. Scanned PDFs and complex table or double-column layouts may need additional processing.")

uploaded_pdf = st.file_uploader("PDF document", type=["pdf"])
question = st.text_area(
    "Question",
    placeholder="What is the main conclusion of this document?",
    height=112,
)

ask_clicked = st.button("Ask document", type="primary", use_container_width=True)

if ask_clicked:
    api_key = load_api_key()
    if uploaded_pdf is None:
        st.error("Choose a PDF before asking a question.")
    elif not question.strip():
        st.error("Enter a question.")
    elif not api_key:
        st.error("DEEPSEEK_API_KEY is not set. Add it to the local .env file before starting the demo.")
    elif top_k > candidate_k:
        st.error("Sources for answer cannot exceed recall candidates.")
    else:
        try:
            with st.spinner("Parsing, retrieving evidence, reranking, and generating an answer..."):
                result = run_question(
                    uploaded_pdf=uploaded_pdf,
                    query=question.strip(),
                    candidate_k=int(candidate_k),
                    top_k=int(top_k),
                    allow_download=allow_download,
                )
        except Exception as exc:
            st.error(f"Could not answer this question: {exc}")
        else:
            document = result["chunking"]
            ocr_pages = result["parsing"]["pages_needing_ocr_review"]
            st.success(
                f"Processed {result['parsing']['page_count']} pages into {document['chunk_count']} chunks."
            )
            if ocr_pages:
                st.warning(f"{ocr_pages} page(s) have little extracted text and may need OCR review.")

            st.subheader("Answer")
            st.markdown(result["answer"]["answer"])
            render_sources(result["answer"]["sources"])

            with st.expander("Run details"):
                st.json(
                    {
                        "embedding_model": result["answer"]["retrieval_method"]["retrieval"]["model"],
                        "rerank_model": result["answer"]["retrieval_method"]["rerank"]["model"],
                        "context_budget": result["answer"]["context_budget"],
                        "token_usage": result["answer"]["usage"],
                    }
                )
