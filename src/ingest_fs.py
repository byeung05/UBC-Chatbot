# src/ingest_fs.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader, WebBaseLoader
)

# -------- Config --------
@dataclass
class IngestConfig:
    paths: Iterable[str | Path]          # files and/or folders
    url: Optional[str] = None            # optional single URL
    chunk_size: int = 1000
    chunk_overlap: int = 100
    fraction: float = 1.0                # take first X% of chunks (0<...<=1]

# -------- Internal: loaders --------
_EXT_LOADERS = {
    ".csv":  lambda p: CSVLoader(file_path=str(p)),
    ".pdf":  lambda p: PyPDFLoader(str(p)),
    ".docx": lambda p: Docx2txtLoader(str(p)),
    ".txt":  lambda p: TextLoader(str(p)),
    ".md":   lambda p: TextLoader(str(p)),
}

def iter_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    for ps in paths:
        p = Path(ps)
        if not p.exists():
            print(f"[warn] Path not found: {p}")
            continue
        if p.is_file():
            yield p
        else:
            for fp in p.rglob("*"):
                if fp.is_file() and not fp.name.startswith("."):
                    yield fp

def load_docs(cfg: IngestConfig) -> List[Document]:
    docs: List[Document] = []
    for f in iter_files(cfg.paths):
        factory = _EXT_LOADERS.get(f.suffix.lower())
        if not factory:
            # print(f"[skip] {f.name}")
            continue
        try:
            docs.extend(factory(f).load())
        except Exception as e:
            print(f"[error] {f.name}: {e}")

    if cfg.url:
        try:
            docs.extend(WebBaseLoader(cfg.url).load())
        except Exception as e:
            print(f"[error] URL load failed: {e}")
    return docs

def chunk(docs: List[Document], cfg: IngestConfig) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
    )
    return splitter.split_documents(docs)

def take_fraction(chunks: List[Document], fraction: float) -> List[Document]:
    if not (0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1].")
    k = max(1, int(len(chunks) * fraction))
    return chunks[:k]

# -------- Public API --------
def ingest(cfg: IngestConfig) -> List[Document]:
    """Load -> chunk -> optionally take first X% of chunks."""
    docs = load_docs(cfg)
    chunks = chunk(docs, cfg)
    if cfg.fraction < 1.0:
        chunks = take_fraction(chunks, cfg.fraction)
    print(f"[ingest] docs={len(docs)} chunks={len(chunks)} (fraction={cfg.fraction:.2f})")
    return chunks
