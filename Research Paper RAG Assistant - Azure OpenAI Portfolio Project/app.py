"""Research Paper RAG Assistant - a small Azure OpenAI + Streamlit portfolio project."""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
INDEX_PATH = STORAGE_DIR / "papers.faiss"
METADATA_PATH = STORAGE_DIR / "chunks.json"
MANIFEST_PATH = STORAGE_DIR / "manifest.json"

CHUNK_SIZE = 1_300  # characters: deliberately small enough for clear citations
CHUNK_OVERLAP = 200
TOP_K = 3
INDEX_VERSION = "1"

PAPER_TITLES = {
    "1706.03762v7.pdf": "Attention Is All You Need",
    "2005.11401v4.pdf": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "2005.14165v4.pdf": "Language Models are Few-Shot Learners",
}

ENVIRONMENT_VARIABLES = {
    "endpoint": "AZURE_OPENAI_ENDPOINT",
    "api_key": "AZURE_OPENAI_API_KEY",
    "api_version": "AZURE_OPENAI_API_VERSION",
    "chat_deployment": "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "embedding_deployment": "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
}

SAMPLE_QUESTIONS = [
    "What are the main components of a RAG model, and how do they interact?",
    "What are the two sub-layers in each encoder layer of the Transformer model?",
    "Explain positional encoding in Transformers and why it is necessary.",
    "What is multi-head attention and why is it beneficial?",
    "What is few-shot learning, and how does GPT-3 implement it during inference?",
]


@dataclass(frozen=True)
class AzureConfig:
    endpoint: str
    api_key: str
    api_version: str
    chat_deployment: str
    embedding_deployment: str


def load_config() -> tuple[AzureConfig | None, list[str]]:
    """Read configuration without ever rendering secret values in the UI."""
    load_dotenv(BASE_DIR / ".env")
    values = {
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "").strip(),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY", "").strip(),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "").strip(),
        "chat_deployment": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip(),
        "embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "").strip(),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        return None, missing
    return AzureConfig(**values), []


def azure_client(config: AzureConfig) -> AzureOpenAI:
    return AzureOpenAI(
        api_key=config.api_key,
        api_version=config.api_version,
        azure_endpoint=config.endpoint,
    )


def source_fingerprint() -> str:
    """Change the fingerprint whenever a source PDF or chunking setting changes."""
    sources = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        stat = pdf_path.stat()
        sources.append({"name": pdf_path.name, "bytes": stat.st_size, "modified": stat.st_mtime_ns})
    payload = {
        "index_version": INDEX_VERSION,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "sources": sources,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def index_is_current() -> bool:
    if not all(path.exists() for path in (INDEX_PATH, METADATA_PATH, MANIFEST_PATH)):
        return False
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["fingerprint"] == source_fingerprint()
    except (OSError, ValueError, KeyError):
        return False


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_into_chunks(text: str) -> list[str]:
    """Split near sentence boundaries while preserving a small overlap."""
    text = clean_text(text)
    if len(text) <= CHUNK_SIZE:
        return [text] if text else []

    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            natural_break = max(text.rfind(". ", start, end), text.rfind("? ", start, end), text.rfind("! ", start, end))
            if natural_break > start + CHUNK_SIZE // 2:
                end = natural_break + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def extract_chunks() -> list[dict[str, Any]]:
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError("No PDFs found. Put the research papers in the data folder.")

    chunks: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        paper_title = PAPER_TITLES.get(pdf_path.name, pdf_path.stem)
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for chunk_number, chunk in enumerate(split_into_chunks(page_text), start=1):
                chunks.append(
                    {
                        "chunk_id": f"{pdf_path.stem}-p{page_number}-c{chunk_number}",
                        "paper_name": paper_title,
                        "file_name": pdf_path.name,
                        "page_number": page_number,
                        "text": chunk,
                    }
                )
    if not chunks:
        raise ValueError("The PDFs did not contain extractable text.")
    return chunks


def embed_texts(client: AzureOpenAI, deployment: str, texts: list[str]) -> np.ndarray:
    """Embed batches to keep indexing reliable for a small portfolio dataset."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 32):
        response = client.embeddings.create(model=deployment, input=texts[start : start + 32])
        vectors.extend(item.embedding for item in response.data)
    array = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(array)
    return array


def build_index(config: AzureConfig) -> tuple[faiss.Index, list[dict[str, Any]]]:
    chunks = extract_chunks()
    vectors = embed_texts(azure_client(config), config.embedding_deployment, [chunk["text"] for chunk in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])  # cosine similarity after vector normalization
    index.add(vectors)

    STORAGE_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    METADATA_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST_PATH.write_text(
        json.dumps({"fingerprint": source_fingerprint(), "chunk_count": len(chunks)}, indent=2), encoding="utf-8"
    )
    return index, chunks


def load_index() -> tuple[faiss.Index, list[dict[str, Any]]]:
    return faiss.read_index(str(INDEX_PATH)), json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def get_or_build_index(config: AzureConfig, force_rebuild: bool = False) -> tuple[faiss.Index, list[dict[str, Any]], bool]:
    if not force_rebuild and index_is_current():
        index, metadata = load_index()
        return index, metadata, False
    index, metadata = build_index(config)
    return index, metadata, True


def retrieve(question: str, config: AzureConfig, index: faiss.Index, metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_vector = embed_texts(azure_client(config), config.embedding_deployment, [question])
    scores, ids = index.search(query_vector, min(TOP_K, len(metadata)))
    results = []
    for score, chunk_index in zip(scores[0], ids[0]):
        if chunk_index >= 0:
            results.append({**metadata[int(chunk_index)], "similarity": float(score)})
    return results


def generate_answer(question: str, contexts: list[dict[str, Any]], config: AzureConfig) -> str:
    formatted_context = "\n\n".join(
        f"SOURCE: [{item['paper_name']}, p. {item['page_number']}]\n{item['text']}" for item in contexts
    )
    system_prompt = """You are a careful research-paper assistant for a beginner in generative AI.
Answer only from the supplied source excerpts. Do not use outside knowledge or invent details.
If the excerpts do not contain enough information, say exactly: "I could not find this in the provided papers."
Explain clearly in 2-4 short paragraphs. Every factual claim must include a source citation in this exact style:
[Paper title, p. number]. Do not cite a source not present in the excerpts."""
    user_prompt = f"""Question: {question}

Retrieved source excerpts:
{formatted_context}"""
    response = azure_client(config).chat.completions.create(
        model=config.chat_deployment,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or "I could not generate an answer. Please try again."


def set_question(question: str) -> None:
    st.session_state.question = question


def index_status() -> str:
    if not index_is_current():
        return "Not built yet"
    try:
        count = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["chunk_count"]
        return f"Ready - {count} chunks"
    except (OSError, ValueError, KeyError):
        return "Needs refresh"


def main() -> None:
    st.set_page_config(page_title="Research Paper RAG Assistant", page_icon="📚", layout="wide")
    st.markdown(
        """<style>
        .block-container {max-width: 1120px; padding-top: 2.5rem;}
        .source-card {border-left: 4px solid #4f46e5; padding: 0.8rem 1rem; background: #f8fafc; margin: 0.5rem 0; border-radius: 0.25rem;}
        </style>""",
        unsafe_allow_html=True,
    )

    config, missing = load_config()
    with st.sidebar:
        st.header("Project status")
        st.write(f"**Index:** {index_status()}")
        if config:
            st.success("Azure configuration loaded")
        else:
            st.warning("Azure configuration needed")
        st.divider()
        st.subheader("How it works")
        st.caption("PDFs -> chunks -> Azure embeddings -> FAISS search -> grounded Azure OpenAI answer")
        refresh = st.button("Refresh index", use_container_width=True, help="Re-embed the papers after they change.")

    st.title("📚 Research Paper RAG Assistant")
    st.write("Ask questions about three landmark AI papers. Answers are grounded in retrieved paper excerpts and include page-level citations.")

    if missing:
        st.info(
            "Add the missing values to a `.env` file using `.env.example`, then restart the app: "
            + ", ".join(f"`{ENVIRONMENT_VARIABLES[item]}`" for item in missing)
        )

    st.subheader("Try a sample question")
    columns = st.columns(2)
    for position, sample in enumerate(SAMPLE_QUESTIONS):
        columns[position % 2].button(sample, key=f"sample_{position}", on_click=set_question, args=(sample,), use_container_width=True)

    st.subheader("Ask the papers")
    st.text_area("Your question", key="question", placeholder="For example: Why does the Transformer use positional encoding?", height=100)
    ask = st.button("Ask question", type="primary")

    if refresh:
        if not config:
            st.error("Azure configuration is required before the index can be built. See `.env.example`.")
        else:
            try:
                with st.spinner("Reading the papers and creating embeddings..."):
                    _, metadata, _ = get_or_build_index(config, force_rebuild=True)
                st.success(f"Index refreshed successfully with {len(metadata)} chunks.")
            except Exception as error:  # Keep provider details out of the interface.
                st.error(f"The index could not be refreshed. Check your Azure endpoint, deployment names, and network access. ({type(error).__name__})")

    if ask:
        question = st.session_state.get("question", "").strip()
        if not question:
            st.warning("Please enter a question first.")
        elif not config:
            st.error("Azure configuration is missing. Copy `.env.example` to `.env`, fill it in, and restart the app.")
        else:
            try:
                with st.spinner("Finding relevant passages and writing a grounded answer..."):
                    index, metadata, was_built = get_or_build_index(config)
                    contexts = retrieve(question, config, index, metadata)
                    answer = generate_answer(question, contexts, config)
                if was_built:
                    st.caption("The local index was created and will be reused on future runs.")
                st.subheader("Answer")
                st.markdown(answer)

                st.subheader("Sources retrieved")
                for item in contexts:
                    st.markdown(
                        f"<div class='source-card'><strong>{item['paper_name']}</strong> - page {item['page_number']} "
                        f"<br><small>Similarity: {item['similarity']:.3f} | {item['chunk_id']}</small></div>",
                        unsafe_allow_html=True,
                    )
                with st.expander("View retrieved excerpts"):
                    for item in contexts:
                        st.markdown(f"**{item['paper_name']} - page {item['page_number']}**")
                        st.write(item["text"])
                        st.divider()
            except Exception as error:  # Provider errors often include sensitive request context.
                st.error(
                    "I could not complete the request. Check your Azure endpoint, API version, deployment names, and network access. "
                    f"({type(error).__name__})"
                )


if __name__ == "__main__":
    main()
