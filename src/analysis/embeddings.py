"""Sentence-BERT embeddings with .npy caching."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    import pandas as pd


class EmbeddingManager:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 128,
        device: str = "mps",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str], cache_path: str | Path | None = None) -> np.ndarray:
        if cache_path:
            cache_path = Path(cache_path)
            if cache_path.exists():
                embs: np.ndarray = np.load(str(cache_path))
                if embs.shape[0] == len(texts):
                    return embs

        model = self._load_model()
        embs = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), embs)

        return embs

    def encode_column(
        self,
        df: pd.DataFrame,
        column: str,
        cache_path: str | Path | None = None,
        fill_value: str = "",
    ) -> np.ndarray:
        texts = df[column].fillna(fill_value).tolist()
        return self.encode(texts, cache_path=cache_path)
