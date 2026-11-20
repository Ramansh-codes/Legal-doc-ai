import faiss
import numpy as np


class VectorIndex:
    """Flat inner-product index. With normalised vectors this is cosine search."""

    def __init__(self, dim: int = None, path: str = None):
        if path:
            self.index = faiss.read_index(path)
        else:
            self.index = faiss.IndexFlatIP(dim)

    def add(self, embeddings: np.ndarray):
        self.index.add(np.ascontiguousarray(embeddings, dtype="float32"))

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def size(self):
        return self.index.ntotal

    def search(self, query_vec: np.ndarray, k=4):
        """Returns [(chunk_index, score), ...] for a single query vector."""
        q = np.ascontiguousarray(query_vec, dtype="float32")
        scores, ids = self.index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
