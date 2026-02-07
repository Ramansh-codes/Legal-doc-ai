from app.embedder import Embedder
from app.index import VectorIndex
from app.llm import generate_answer
import json


def main():
    # 🔹 Load chunks
    with open("data/chunks.json", "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    # 🔹 Load FAISS index from file
    index = VectorIndex(path="data/legal.index")

    # 🔹 Embedder for question
    embedder = Embedder()

    print("Legal Q&A system ready. Type 'exit' to quit.")

    while True:
        question = input("\nAsk: ").strip()
        if question.lower() in ("exit", "quit"):
            break

        # 🔹 Embed question
        q_vec = embedder.embed([question])

        # 🔹 Search in FAISS
        _, ids = index.search(q_vec, k=3)

        # ❗ FIX: ids is 2D array
        retrieved_chunks = [all_chunks[i] for i in ids[0]]

        # 🔹 Ask LLM
        answer = generate_answer(question, retrieved_chunks)

        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    main()
