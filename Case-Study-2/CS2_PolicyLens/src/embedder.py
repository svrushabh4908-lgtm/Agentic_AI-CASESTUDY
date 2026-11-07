"""
Stage 5: Embed (src/embedder.py)
Converts text chunks into L2-normalized dense vectors in batches using Gemini's
`gemini-embedding-001` model, with exponential backoff retries and rate-limit handling.
"""

import math
import os
import time
from typing import List, Tuple
from dotenv import load_dotenv
from google import genai

load_dotenv()


def _l2_normalize(vec: List[float]) -> List[float]:
    """L2-normalizes a vector so cosine similarity equals dot product."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class GeminiBatchEmbedder:
    """Batch embedding wrapper around google-genai with retry & rate-limit backoff."""

    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        batch_size: int = 64,
        max_retries: int = 5,
        base_backoff: float = 2.0,
        normalize: bool = True,
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment or .env file.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.normalize = normalize
        self.total_api_calls = 0
        self.total_embedded_chunks = 0

    def embed_texts(self, texts: List[str]) -> Tuple[List[List[float]], float]:
        """
        Embeds a list of text strings in batches.
        Returns (list_of_vectors, elapsed_seconds).
        """
        if not texts:
            return [], 0.0

        all_vectors: List[List[float]] = []
        t0 = time.perf_counter()

        for start_idx in range(0, len(texts), self.batch_size):
            batch = texts[start_idx : start_idx + self.batch_size]
            attempt = 0
            while True:
                try:
                    res = self.client.models.embed_content(
                        model=self.model_name,
                        contents=batch,
                    )
                    self.total_api_calls += 1
                    for emb in res.embeddings:
                        vec = list(emb.values)
                        if self.normalize:
                            vec = _l2_normalize(vec)
                        all_vectors.append(vec)
                    self.total_embedded_chunks += len(batch)
                    break
                except Exception as exc:
                    attempt += 1
                    if attempt > self.max_retries:
                        raise RuntimeError(
                            f"Embedding batch failed after {self.max_retries} retries: {exc}"
                        ) from exc
                    sleep_sec = self.base_backoff * (2 ** (attempt - 1))
                    print(
                        f"  [WARN] Rate limit or transient error ({exc}). "
                        f"Retrying batch in {sleep_sec:.1f}s (attempt {attempt}/{self.max_retries})..."
                    )
                    time.sleep(sleep_sec)

        elapsed = time.perf_counter() - t0
        return all_vectors, elapsed

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single search query string."""
        vecs, _ = self.embed_texts([query])
        return vecs[0]