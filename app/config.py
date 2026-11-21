"""Central configuration. Everything is overridable via environment variables,
so the same code runs locally, in Docker and on AWS without edits."""
import os

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_PATH = os.path.join(DATA_DIR, "legal.index")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "350"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "4"))

LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini")
# Retrieval score below which we skip the LLM and answer "not found".
# 0.0 = disabled. Tune it with the evaluation set (Nov 23).
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.0"))
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
