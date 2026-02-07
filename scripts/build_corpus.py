from app.data_loader import load_legal_dataset
from app.cleaner import clean_text
from app.chunker import chunk_text
from app.embedder import Embedder
from app.index import VectorIndex
import json
import os


def main():
    data = load_legal_dataset()
    all_chunks = []

    # 🔹 FOR NOW: only first document
    raw = data[0]["judgement"]
    clean = clean_text(raw)
    chunks = chunk_text(clean)
    all_chunks.extend(chunks)

    print("Total chunks:", len(all_chunks))

    # 🔹 Embedding
    embedder = Embedder()
    embeddings = embedder.embed(all_chunks)

    # 🔹 Build FAISS index
    index = VectorIndex(dim=embeddings.shape[1])
    index.add(embeddings)

    # 🔹 Ensure data folder exists
    os.makedirs("data", exist_ok=True)

    # 🔹 Save index
    index.save("data/legal.index")
    print("FAISS index saved at data/legal.index")

    # 🔹 Save chunks (VERY IMPORTANT for mapping later)
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("Chunks saved to data/chunks.json")


if __name__ == "__main__":
    main()
