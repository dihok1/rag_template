"""Build vector index from knowledge base directory."""
import json
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

try:
    import faiss
except ImportError:
    faiss = None

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    INDEX_PATH,
    KNOWLEDGE_BASE_PATH,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
)
from app.rag.text_cleaning import clean_text, should_skip_path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _LANGCHAIN_SPLITTER = True
except ImportError:
    _LANGCHAIN_SPLITTER = False


def _read_file(path: Path, encoding: str = "utf-8") -> str:
    try:
        return path.read_text(encoding=encoding)
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace")


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Разбиение по границам параграфов где возможно, иначе по символам с перекрытием.
    Сохраняет целостность абзацев и улучшает качество эмбеддингов.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text] if text else []

    chunks = []
    current = []
    current_len = 0

    for i, para in enumerate(paragraphs):
        add_len = len(para) + 2  # +2 for \n\n
        if current_len + add_len <= chunk_size:
            current.append(para)
            current_len += add_len
        else:
            if current:
                chunks.append("\n\n".join(current))
            # overlap: оставляем последний параграф в начале следующего чанка
            if overlap > 0 and current:
                overlap_paras = []
                overlap_len = 0
                for p in reversed(current):
                    if overlap_len + len(p) + 2 <= overlap:
                        overlap_paras.append(p)
                        overlap_len += len(p) + 2
                    else:
                        break
                overlap_paras.reverse()
                current = overlap_paras
                current_len = sum(len(p) + 2 for p in current)
            else:
                current = []
                current_len = 0
            if current_len + add_len <= chunk_size:
                current.append(para)
                current_len += add_len
            else:
                # параграф больше chunk_size — режем по символам
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunk = para[start:end]
                    if chunk.strip():
                        chunks.append(chunk.strip())
                    start = end - overlap

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _collect_documents(base_path: Path) -> list[tuple[Path, str]]:
    """Recursively collect .md and .txt files (skip macOS ._*); return (path, cleaned content)."""
    out: list[tuple[Path, str]] = []
    for ext in (".md", ".txt"):
        for path in base_path.rglob(f"*{ext}"):
            if not path.is_file() or should_skip_path(path):
                continue
            raw = _read_file(path)
            content = clean_text(raw)
            if content.strip():
                out.append((path, content))
    return out


def _get_embeddings(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    """Batch embed texts (OpenAI allows batch)."""
    if not texts:
        return []
    batch_size = 100
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(input=batch, model=model)
        for e in resp.data:
            all_embeddings.append(e.embedding)
    return all_embeddings


def build_index(
    knowledge_base_path: Path | None = None,
    index_path: Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> None:
    """Index all .md/.txt under knowledge_base_path; save FAISS index + metadata to index_path."""
    if faiss is None:
        raise RuntimeError("faiss-cpu is required for indexing. Install: pip install faiss-cpu")

    kb = knowledge_base_path or KNOWLEDGE_BASE_PATH
    idx_path = index_path or INDEX_PATH
    cs = chunk_size if chunk_size is not None else CHUNK_SIZE
    co = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")

    kb = kb.resolve()
    if not kb.is_dir():
        raise FileNotFoundError(f"Knowledge base path is not a directory: {kb}")

    idx_path = idx_path.resolve()
    idx_path.mkdir(parents=True, exist_ok=True)

    documents = _collect_documents(kb)
    if not documents:
        raise ValueError(f"No .md or .txt files found under {kb}")

    chunks: list[dict[str, Any]] = []
    if _LANGCHAIN_SPLITTER:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cs,
            chunk_overlap=co,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
    for path, content in documents:
        rel_path = path.relative_to(kb) if path.is_relative_to(kb) else path
        if _LANGCHAIN_SPLITTER:
            chunk_texts = splitter.split_text(content)
        else:
            chunk_texts = _chunk_text(content, cs, co)
        for i, chunk_text in enumerate(chunk_texts):
            if not chunk_text.strip():
                continue
            chunks.append({
                "text": chunk_text.strip(),
                "source_path": str(rel_path),
                "chunk_index": i,
            })

    if not chunks:
        raise ValueError("No text chunks produced from documents")

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    texts = [c["text"] for c in chunks]
    embeddings = _get_embeddings(client, texts, OPENAI_EMBEDDING_MODEL)
    matrix = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(matrix.shape[1])  # inner product; embeddings are normalized
    faiss.normalize_L2(matrix)
    index.add(matrix)

    faiss.write_index(index, str(idx_path / "index.faiss"))
    meta_path = idx_path / "metadata.json"
    meta_path.write_text(
        json.dumps({"chunks": chunks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Index built: {len(chunks)} chunks, saved to {idx_path}")


if __name__ == "__main__":
    build_index()
