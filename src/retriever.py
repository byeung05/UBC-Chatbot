"""
Hybrid retriever that blends dense (Gemini) + sparse (TF-IDF) with alpha.
"""

from langchain_core.documents import Document
from .config import SETTINGS
from .embeddings import make_embedder, l2_normalize

class PineconeHybridRetriever:
    def __init__(self, index, vectorizer, alpha: float = 0.6, top_k: int = 10, flt=None):
        self.index = index
        self.vectorizer = vectorizer
        self.alpha = alpha
        self.top_k = top_k
        self.flt = flt
        self.emb = make_embedder()

    def invoke(self, query: str):
        # dense part
        q_dense = l2_normalize(self.emb.embed_query(query))
        q_dense = [self.alpha * x for x in q_dense]
        # sparse part
        q_row = self.vectorizer.transform([query])
        q_idx = q_row.indices.tolist()
        q_val = [(1 - self.alpha) * float(v) for v in q_row.data.tolist()]
        # search
        res = self.index.query(
            vector=q_dense,
            sparse_vector={"indices": q_idx, "values": q_val},
            top_k=self.top_k,
            include_metadata=True,
            filter=self.flt,
            namespace=SETTINGS.NAMESPACE
        )
        docs = []
        for m in res.get("matches", []):
            md = m.get("metadata", {}) or {}
            txt = md.pop("text", "")
            docs.append(Document(page_content=txt, metadata=md | {"_score": m.get("score")}))
        return docs

