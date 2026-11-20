import json

from app import config
from app.embedder import Embedder
from app.index import VectorIndex


class Retriever:
    """Loads the saved index + chunk metadata once; the API reuses one instance."""

    def __init__(self, index_path=config.INDEX_PATH, chunks_path=config.CHUNKS_PATH):
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.index = VectorIndex(path=index_path)
        self.embedder = Embedder()

        if self.index.size() != len(self.chunks):
            raise RuntimeError(
                f"Index has {self.index.size()} vectors but chunks file has "
                f"{len(self.chunks)}; rebuild the corpus."
            )

    def retrieve(self, question, k=config.TOP_K):
        q_vec = self.embedder.embed([question])
        results = []
        for idx, score in self.index.search(q_vec, k):
            item = dict(self.chunks[idx])
            item["score"] = score
            results.append(item)
        return results
