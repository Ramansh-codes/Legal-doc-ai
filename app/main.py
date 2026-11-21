"""FastAPI service.  Run:  uvicorn app.main:app --reload
Docs UI:  http://127.0.0.1:8000/docs"""
import logging
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import config, llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("legal_qa")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(config.TOP_K, ge=1, le=10)


class Source(BaseModel):
    ref: int  # matches the [n] citations in the answer
    doc_id: int
    chunk_id: int
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    found: bool
    sources: List[Source]
    latency_ms: int


def _default_retriever():
    from app.retriever import Retriever  # heavy import (torch); keep it lazy

    return Retriever()


def create_app(retriever_factory=_default_retriever, min_score=config.MIN_SCORE):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.retriever = retriever_factory()  # load model + index once
        log.info("Retriever ready: %d chunks", len(app.state.retriever.chunks))
        yield

    app = FastAPI(title="Legal Doc Q&A", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.ALLOWED_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": len(app.state.retriever.chunks)}

    # Plain `def` (not async): FastAPI runs it in a worker thread, so the
    # blocking embedding + LLM calls don't freeze the server.
    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest):
        t0 = time.perf_counter()
        chunks = app.state.retriever.retrieve(req.question, req.top_k)

        if not chunks or chunks[0]["score"] < min_score:
            answer, used = llm.NOT_FOUND, []
        else:
            try:
                answer = llm.generate_answer(req.question, chunks)
            except Exception:
                log.exception("LLM call failed")
                raise HTTPException(502, "The language model request failed. Please try again.")
            used = chunks

        latency_ms = int((time.perf_counter() - t0) * 1000)
        found = llm.NOT_FOUND not in answer
        log.info("query len=%d k=%d top_score=%.3f found=%s ms=%d",
                 len(req.question), req.top_k,
                 chunks[0]["score"] if chunks else -1.0, found, latency_ms)

        return QueryResponse(
            answer=answer,
            found=found,
            latency_ms=latency_ms,
            sources=[
                Source(ref=n, doc_id=c["doc_id"], chunk_id=c["chunk_id"],
                       score=round(c["score"], 4), snippet=c["text"][:300])
                for n, c in enumerate(used, 1)
            ],
        )

    return app


app = create_app()
