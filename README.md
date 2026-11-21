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
# create a file named .env in the project root containing:
#   GEMINI_API_KEY=your_key_here
```

## Run
```bash
python -m scripts.build_corpus --max-docs 300   # build index (omit flag for all docs)
python -m scripts.query                         # interactive Q&A (CLI)
uvicorn app.main:app --reload                   # API: http://127.0.0.1:8000/docs
python -m pytest                                # tests
```

## API
```bash
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
     -d '{"question": "What did the court decide about the lease?", "top_k": 4}'
```
Returns `answer`, `found`, `latency_ms` and `sources` (each with `ref`, `doc_id`, `chunk_id`, `score`, `snippet`).

Settings (environment variables / `.env`): `GEMINI_API_KEY`, `GEMINI_MODEL`, `LLM_BACKEND`,
`TOP_K`, `MIN_SCORE`, `ALLOWED_ORIGINS`, `DATA_DIR`, `EMBED_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`.

## Layout
```text
app/        config, data_loader, cleaner, chunker, embedder, index, retriever, llm, main (FastAPI)
scripts/    build_corpus.py, query.py
tests/      unit tests
data/       generated index + chunks (git-ignored)
```

## Status
- [x] Retrieval pipeline with chunk metadata and citations
- [x] FastAPI service (`/query`, `/health`)
- [ ] Evaluation (hit@k, latency)
- [ ] Docker + AWS deployment, CI/CD
