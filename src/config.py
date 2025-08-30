"""
Centralized config & constants.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # read .env if present

@dataclass(frozen=True)
class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
    BASE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "ubc-grades")
    HYBRID_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "ubc-grades") + "-hybrid"
    NAMESPACE: str = "ubc-grades"
    FRACTION: float = 0.10  # index first 10% initially
    UPSERT_BATCH: int = 100
    EMBED_INIT_BATCH: int = 64
    EMBED_MIN_BATCH: int = 8
    EMBED_MAX_RETRIES: int = 6
    TFIDF_PATH: str = "tfidf.joblib"  # persisted vectorizer

SETTINGS = Settings()

def require_env():
    assert SETTINGS.GEMINI_API_KEY, "Missing GEMINI_API_KEY"
    assert SETTINGS.PINECONE_API_KEY, "Missing PINECONE_API_KEY"
