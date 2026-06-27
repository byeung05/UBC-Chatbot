"""
Hybrid retriever that blends dense (Gemini) + sparse (TF-IDF) with alpha.
- Skips sparse_vector when empty to avoid Pinecone 400s.
- Guards against empty/whitespace queries.
"""

from langchain_core.documents import Document
from .config import SETTINGS
from .embeddings import make_embedder, l2_normalize

class PineconeHybridRetriever:
    def __init__(self, index, vectorizer, alpha: float = 0.6, top_k: int = 10, flt=None,
                 min_query_chars: int = 1,  # set to 3 if you want stricter gating
                 allow_dense_fallback_for_short: bool = True):
        self.index = index
        self.vectorizer = vectorizer
        self.alpha = float(alpha)
        self.top_k = int(top_k)
        self.flt = flt
        self.emb = make_embedder()
        self.min_query_chars = int(min_query_chars)
        self.allow_dense_fallback_for_short = bool(allow_dense_fallback_for_short)

    def _make_dense(self, query: str):
        # Embed + L2 normalize per your pipeline
        vec = self.emb.embed_query(query) or []
        # Defensive: if the embedder ever returns a zero vector, skip normalize scaling
        if not vec:
            return None
        v = l2_normalize(vec)
        # Apply alpha weighting for hybrid score blending
        return [self.alpha * x for x in v]

    def _make_sparse(self, query: str):
        # TF-IDF/BM25 → CSR row
        row = self.vectorizer.transform([query])
        # row.indices, row.data are numpy arrays; convert to py lists
        idx = row.indices.tolist()
        if not idx:
            return None
        val = row.data.tolist()
        # Apply (1 - alpha) scaling to sparse weights
        val = [(1.0 - self.alpha) * float(v) for v in val]
        return {"indices": idx, "values": val}

    def invoke(self, query: str):
        if query is None or len(query.strip()) < self.min_query_chars:
            # Two reasonable options:
            # 1) Return [] to indicate "nothing to retrieve"
            # 2) If you prefer, do dense-only fallback for ultra-short inputs
            if not self.allow_dense_fallback_for_short:
                return []
            # else: proceed to dense-only fallback below

        q_dense = self._make_dense(query)
        q_sparse = self._make_sparse(query)

        # Build query kwargs conditionally to avoid empty sparse_vector error
        q_kwargs = {
            "top_k": self.top_k,
            "include_metadata": True,
            "namespace": SETTINGS.NAMESPACE
        }
        if self.flt is not None:
            q_kwargs["filter"] = self.flt
        if q_dense is not None:
            q_kwargs["vector"] = q_dense
        # Only include sparse_vector if we have at least one index
        if q_sparse is not None and q_sparse.get("indices"):
            q_kwargs["sparse_vector"] = q_sparse

        # If both signals are missing (very unlikely), return empty
        if "vector" not in q_kwargs and "sparse_vector" not in q_kwargs:
            print("[retriever] Skipping Pinecone: no dense or sparse vector built")

            return []

            # in PineconeHybridRetriever.invoke, right before self.index.query(**q_kwargs)
        print("[retriever] Query kwargs:", {k: (len(v) if k in ("vector",) else v) for k, v in q_kwargs.items()})
        res = self.index.query(**q_kwargs)

        docs = []
        for m in res.get("matches", []):
            md = m.get("metadata", {}) or {}
            txt = md.pop("text", "")
            docs.append(
                Document(
                    page_content=txt,
                    metadata=md | {"_score": m.get("score")}
                )
            )
        return docs
