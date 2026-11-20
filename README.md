# Legal Document Q&A (RAG)

Retrieval-Augmented Generation over legal judgments: documents are cleaned,
chunked, embedded (Sentence Transformers), indexed in FAISS, and answered by
Gemini using only the retrieved passages, with numbered source citations.

## Pipeline
1. Load judgments (Hugging Face dataset) and clean text
2. Chunk into overlapping word windows (metadata: doc_id, chunk_id)
3. Embed with `all-MiniLM-L6-v2` (normalised vectors)
4. Store in a FAISS inner-product (cosine) index
5. At query time: embed question -> top-k chunks -> Gemini answers with citations `[1]`, `[2]`

## Setup (Linux / macOS / Windows)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then put your GEMINI_API_KEY in .env
```

## Run
```bash
python -m scripts.build_corpus --max-docs 300   # build index (omit flag for all docs)
python -m scripts.query                         # interactive Q&A
python -m pytest                                # tests
```

## Layout
```text
app/        config, data_loader, cleaner, chunker, embedder, index, retriever, llm
scripts/    build_corpus.py, query.py
tests/      unit tests
data/       generated index + chunks (git-ignored)
```

## Status
- [x] Retrieval pipeline with chunk metadata and citations
- [ ] FastAPI service (`/query`, `/health`)
- [ ] Evaluation (hit@k, latency)
- [ ] Docker + AWS deployment, CI/CD
