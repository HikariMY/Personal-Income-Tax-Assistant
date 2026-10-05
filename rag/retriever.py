"""Sentence-embedding + FAISS vector search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import faiss
import numpy as np

from rag.loader import Chunk

EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"

# E5 models expect these prefixes to distinguish queries from passages.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

EmbedFn = Callable[[Sequence[str]], np.ndarray]


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def load_embedder(model_name: str = EMBED_MODEL_NAME) -> EmbedFn:
    """Return a function that maps texts to L2-normalized float32 vectors."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")

    def embed(texts: Sequence[str]) -> np.ndarray:
        vectors = model.encode(
            list(texts), batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype="float32")

    return embed


def passage_text(chunk: Chunk) -> str:
    """Text used for embedding: title/heading context + chunk body."""
    return f"{PASSAGE_PREFIX}{chunk.title} > {chunk.heading}\n{chunk.text}"


class VectorStore:
    """FAISS inner-product index over normalized vectors (= cosine similarity)."""

    def __init__(self, chunks: Sequence[Chunk], embed: EmbedFn):
        if not chunks:
            raise ValueError("VectorStore needs at least one chunk")
        self._chunks = tuple(chunks)
        self._embed = embed
        vectors = embed([passage_text(c) for c in self._chunks])
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int = 4) -> list[SearchResult]:
        if not query.strip():
            return []
        k = max(1, min(k, len(self._chunks)))
        query_vector = self._embed([f"{QUERY_PREFIX}{query}"])
        scores, ids = self._index.search(query_vector, k)
        return [
            SearchResult(chunk=self._chunks[i], score=float(s))
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]
