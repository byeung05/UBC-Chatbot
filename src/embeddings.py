"""
Gemini embeddings + robust batching with retries and L2 normalization.
"""

import math, time, random
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .config import SETTINGS

def make_embedder() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=SETTINGS.GEMINI_API_KEY
    )

def l2_normalize(vec: List[float]) -> List[float]:
    n = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v / n for v in vec]

def adaptive_embed_documents(
    emb: GoogleGenerativeAIEmbeddings,
    texts: List[str],
    init_batch: int = SETTINGS.EMBED_INIT_BATCH,
    min_batch: int = SETTINGS.EMBED_MIN_BATCH,
    max_retries: int = SETTINGS.EMBED_MAX_RETRIES,
    base_sleep: float = 1.0,
    jitter: float = 0.25,
    progress_every: int = 1000,
    normalize: bool = True,
) -> List[List[float]]:
    """
    Embed texts with adaptive batch sizing & retries to avoid 504/429/500/503.
    """
    out: List[List[float]] = []
    i, B, N = 0, init_batch, len(texts)
    while i < N:
        batch = texts[i : min(i+B, N)]
        attempt = 0
        while True:
            try:
                vecs = emb.embed_documents(batch)
                if normalize:
                    vecs = [l2_normalize(v) for v in vecs]
                out.extend(vecs)
                i += len(batch)
                if i % progress_every == 0 or i == N:
                    print(f"[emb] {i}/{N} (batch={B})")
                if B < init_batch:
                    B = min(init_batch, B*2)  # gentle ramp back up
                break
            except Exception as e:
                err = str(e).lower()
                transient = any(t in err for t in ["504", "deadline", "timeout", "503", "500", "429"])
                attempt += 1
                if transient and B > min_batch:
                    B = max(min_batch, B // 2)
                    print(f"[warn] {e} -> reduce batch to {B} and retry…")
                    time.sleep(base_sleep + random.random()*jitter)
                    continue
                if transient and attempt <= max_retries:
                    sleep = base_sleep*(2**(attempt-1)) + random.random()*jitter
                    print(f"[warn] {e} -> retry {attempt}/{max_retries} in {sleep:.1f}s…")
                    time.sleep(sleep)
                    continue
                raise
    return out
