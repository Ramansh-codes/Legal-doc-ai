import faiss
import numpy as np

class VectorIndex:
    def __init__(self, dim: int = None, path: str = None):
        if path:
            self.index = faiss.read_index(path)
        else:
            self.index = faiss.IndexFlatL2(dim)

    def add(self, embeddings: np.ndarray):
        self.index.add(embeddings)

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def size(self):
        return self.index.ntotal

    def search(self, query_vec: np.ndarray, k=3):
        # returns full FAISS output (2D arrays)
        return self.index.search(query_vec, k)
