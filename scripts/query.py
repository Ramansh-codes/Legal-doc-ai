from app.llm import generate_answer
from app.retriever import Retriever


def main():
    retriever = Retriever()
    print("Legal Q&A system ready. Type 'exit' to quit.")

    while True:
        question = input("\nAsk: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        chunks = retriever.retrieve(question)
        answer = generate_answer(question, chunks)

        print("\nAnswer:\n")
        print(answer)
        print("\nSources:")
        for n, c in enumerate(chunks, 1):
            print(f"  [{n}] doc {c['doc_id']} chunk {c['chunk_id']} (score {c['score']:.3f})")


if __name__ == "__main__":
    main()
