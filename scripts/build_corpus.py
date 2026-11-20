"""Build the FAISS index + chunk metadata from the dataset.

Usage:
    python -m scripts.build_corpus --max-docs 300     # quick dev build (Indian cases)
    python -m scripts.build_corpus                    # all Indian cases
    python -m scripts.build_corpus --source ""        # include UK cases too
"""
import argparse
import json
import os
import time

import faiss

from app import config
from app.chunker import chunk_document
from app.cleaner import clean_text
from app.corpus import select_doc_ids
from app.data_loader import load_legal_dataset
from app.embedder import Embedder
from app.index import VectorIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--source", default="IN-Abs",
                        help='dataset_name to keep ("" = keep everything)')
    args = parser.parse_args()

    data = load_legal_dataset()
    print("Dataset columns:", data.column_names, "| rows:", len(data))

    doc_ids = select_doc_ids(data["dataset_name"], args.source or None, args.max_docs)
    n_docs = len(doc_ids)
    all_chunks = []
    for doc_id in doc_ids:
        row = data[doc_id]
        text = clean_text(row["judgement"])
        all_chunks.extend(
            chunk_document(
                doc_id,
                text,
                config.CHUNK_SIZE,
                config.CHUNK_OVERLAP,
                source=str(row.get("dataset_name", "")),
            )
        )
    print(f"Documents: {n_docs} | Chunks: {len(all_chunks)}")

    t0 = time.time()
    embedder = Embedder()
    embeddings = embedder.embed([c["text"] for c in all_chunks], show_progress=True)
    print(f"Embedded in {time.time() - t0:.0f}s")

    index = VectorIndex(dim=embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    index.save(config.INDEX_PATH)
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    print(f"Saved {index.size()} vectors -> {config.INDEX_PATH}, {config.CHUNKS_PATH}")


if __name__ == "__main__":
    main()
