# src/ingest_fs.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader, WebBaseLoader
)

# Map extensions to loaders (extend as needed)
EXT_LOADERS = {
    ".csv":  lambda p: CSVLoader(file_path=str(p)),
    ".pdf":  lambda p: PyPDFLoader(str(p)),
    ".docx": lambda p: Docx2txtLoader(str(p)),
    ".txt":  lambda p: TextLoader(str(p)),
    ".md":   lambda p: TextLoader(str(p)),
}

def _iter_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    """Yield all files from a list of files/dirs; recursive for dirs."""
    for ps in paths:
        p = Path(ps)
        if not p.exists():
            print(f"[warn] Path not found: {p}")
            continue
        if p.is_file():
            yield p
        else:
            # recursive; skip hidden/system files
            for fp in p.rglob("*"):
                if fp.is_file() and not fp.name.startswith("."):
                    yield fp

def load_docs_from_paths(
    paths: Iterable[str | Path],
    url: Optional[str] = None
) -> List[Document]:
    """Load docs from local files (csv/pdf/docx/txt/md) + optional single URL."""
    docs: List[Document] = []
    for f in _iter_files(paths):
        loader_factory = EXT_LOADERS.get(f.suffix.lower())
        if not loader_factory:
            # Uncomment to see skipped types:
            # print(f"[skip] {f.name}")
            continue
        try:
            loader = loader_factory(f)
            docs.extend(loader.load())
        except Exception as e:
            print(f"[error] {f.name}: {e}")

    if url:
        try:
            docs.extend(WebBaseLoader(url).load())
        except Exception as e:
            print(f"[error] URL load failed: {e}")

    return docs

def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> List[Document]:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def take_first_fraction(
    chunks: List[Document],
    fraction: float = 1.0
) -> List[Document]:
    """Keep the FIRST fraction of chunks for fast smoke tests."""
    if not (0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1].")
    k = max(1, int(len(chunks) * fraction))
    return chunks[:k]

