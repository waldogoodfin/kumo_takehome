from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.config import get_settings
from app.state import get_data_bundle
from app.services.llm import get_llm_client


@dataclass
class EmbeddingIndex:
    matrix: np.ndarray
    ids: np.ndarray


class EmbeddingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm_client = get_llm_client()
        self._index: Optional[EmbeddingIndex] = None

    def enabled(self) -> bool:
        return self.settings.enable_embeddings and self.llm_client.enabled

    def ensure_index(self) -> None:
        if not self.enabled() or self._index is not None:
            return

        bundle = get_data_bundle()
        articles = bundle.articles
        sample = articles.sample(
            n=min(len(articles), self.settings.embedding_sample_size),
            random_state=123,
        )

        texts = [
            f"{row['prod_name']} | {row['product_type_name']} | {row['department_name']}"
            for _, row in sample.iterrows()
        ]

        embeddings = self._embed_texts(texts)
        if embeddings is None:
            return

        matrix = np.array(embeddings, dtype="float32")
        ids = sample['article_id'].to_numpy(dtype="int64")
        self._index = EmbeddingIndex(matrix=matrix, ids=ids)

    def _embed_texts(self, texts: list[str]) -> Optional[list[list[float]]]:
        if not self.llm_client.enabled:
            return None
        try:
            client = self.llm_client.get_client()
        except RuntimeError:
            return None

        model = self.settings.embedding_model
        embeddings: list[list[float]] = []
        batch_size = self.settings.embedding_batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(input=batch, model=model)
            for item in response.data:
                embeddings.append(item.embedding)
        return embeddings

    def query(self, text: str, top_k: int = 10) -> list[tuple[int, float]]:
        self.ensure_index()
        if not self.enabled() or self._index is None:
            return []

        query_embedding = self._embed_texts([text])
        if not query_embedding:
            return []

        query_vec = np.array(query_embedding, dtype="float32")
        matrix = self._index.matrix
        query_norm = np.linalg.norm(query_vec, axis=1)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        similarities = (matrix @ query_vec.T).flatten() / (matrix_norms * query_norm)
        top_indices = similarities.argsort()[::-1][:top_k]
        return [
            (int(self._index.ids[idx]), float(similarities[idx]))
            for idx in top_indices
        ]


_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
