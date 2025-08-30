"""
Build a Pinecone DOTPRODUCT index and upsert hybrid vectors (dense + sparse).
- Uses first FRACTION of chunks initially (10%); set to 1.0 for full.
- Requires `document_chunks` list to be present in the caller.
"""

from typing import List
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from .config import SETTINGS, require_env
from .embeddings import make_embedder, adaptive_embed_documents
from .tfidf import build_tfidf, save_vectorizer

def build_and_upsert(document_chunks: List[Document]):
    require_env()

    # --- take first FRACTION in order ---
    n_total = len(document_chunks)
    k = max(1, int(n_total * SETTINGS.FRACTION))
    idxs = list(range(k))
    print(f"[index] using first {k}/{n_total} chunks (fraction={SETTINGS.FRACTION:.2f})")

    chunks = [document_chunks[i] for i in idxs]
    texts = [d.page_content for d in chunks]

    # --- sparse (TF-IDF) ---
    vectorizer, X = build_tfidf(texts)
    save_vectorizer(vectorizer)

    # --- Pinecone index (DOTPRODUCT for hybrid) ---
    emb = make_embedder()
    dim = len(emb.embed_query("dimension check"))

    pc = Pinecone(api_key=SETTINGS.PINECONE_API_KEY)
    existing = [x["name"] for x in pc.list_indexes()]
    if SETTINGS.HYBRID_INDEX_NAME not in existing:
        pc.create_index(
            name=SETTINGS.HYBRID_INDEX_NAME,
            dimension=dim,
            metric="dotproduct",  # REQUIRED for dense+sparse hybrid
            spec=ServerlessSpec(
                cloud=SETTINGS.PINECONE_CLOUD,
                region=SETTINGS.PINECONE_REGION
            ),
        )
    index = pc.Index(SETTINGS.HYBRID_INDEX_NAME)

    # --- dense embeddings (batched) ---
    dense_vecs = adaptive_embed_documents(emb, texts)

    # --- upsert hybrid ---
    metas = [d.metadata | {"text": d.page_content} for d in chunks]
    buf = []
    for j, i_doc in enumerate(idxs):
        row = X.getrow(j)
        s_idx = row.indices.tolist()
        s_val = row.data.astype("float32").tolist()
        buf.append({
            "id": f"{SETTINGS.NAMESPACE}-{i_doc}",
            "values": dense_vecs[j],
            "sparse_values": {"indices": s_idx, "values": s_val},
            "metadata": metas[j],
        })
        if len(buf) == SETTINGS.UPSERT_BATCH:
            index.upsert(vectors=buf, namespace=SETTINGS.NAMESPACE)
            buf = []
    if buf:
        index.upsert(vectors=buf, namespace=SETTINGS.NAMESPACE)

    print(f"[index] upserted {len(idxs)} vectors into '{SETTINGS.HYBRID_INDEX_NAME}' ns='{SETTINGS.NAMESPACE}'")
