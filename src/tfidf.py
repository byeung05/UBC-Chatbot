"""
TF-IDF vectorizer (scikit-learn). No NLTK; persists to disk for reuse.
"""

from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from .config import SETTINGS

def build_tfidf(texts: List[str]) -> Tuple[TfidfVectorizer, "scipy.sparse.csr_matrix"]:
    token_pattern = r"[A-Za-z0-9]+"  # keeps CPSC221 / 2023W2 tokens
    vectorizer = TfidfVectorizer(
        token_pattern=token_pattern,
        lowercase=True,
        stop_words="english",
        min_df=2
    )
    X = vectorizer.fit_transform(texts)
    print(f"[tfidf] built: {X.shape[0]} docs, vocab~{X.shape[1]}")
    return vectorizer, X

def save_vectorizer(vectorizer: TfidfVectorizer, path: str = SETTINGS.TFIDF_PATH):
    joblib.dump(vectorizer, path)
    print(f"[tfidf] saved -> {path}")

def load_vectorizer(path: str = SETTINGS.TFIDF_PATH) -> TfidfVectorizer:
    vec = joblib.load(path)
    print(f"[tfidf] loaded <- {path}")
    return vec
