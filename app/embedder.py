import numpy as np
from sentence_transformers import SentenceTransformer

from app import config


class Embedder:
    def __init__(self, model_name=config.EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts, show_progress=False, batch_size=64):
        """texts: List[str] -> float32 array (n, dim), L2-normalised so that
        inner product == cosine similarity."""
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return np.asarray(vecs, dtype="float32")
