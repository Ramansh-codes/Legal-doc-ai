from app import config

NOT_FOUND = "Not found in the provided documents."

_model = None


def _get_model():
    """Configure Gemini lazily so importing this module never needs a key."""
    global _model
    if _model is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set (put it in .env).")
        import google.generativeai as genai

        genai.configure(api_key=config.GEMINI_API_KEY)
        _model = genai.GenerativeModel(config.GEMINI_MODEL)
    return _model


def build_prompt(question, chunks):
    """chunks: list of dicts with 'text' (and optionally doc_id/chunk_id)."""
    context = "\n\n".join(f"[{n}] {c['text']}" for n, c in enumerate(chunks, 1))
    return f"""You are a legal assistant.

Answer the question ONLY using the numbered context passages below.
After each claim, cite the passage(s) it came from like [1] or [2][3].
If the answer is not present in the passages, reply exactly: "{NOT_FOUND}"

Context:
{context}

Question:
{question}

Answer:
"""


def _gemini_generate(prompt):
    return _get_model().generate_content(prompt).text.strip()


# To add another backend (e.g. a self-hosted open model), write a function
# prompt -> str and register it here, then set LLM_BACKEND in the environment.
BACKENDS = {"gemini": _gemini_generate}


def generate_answer(question, chunks):
    try:
        backend = BACKENDS[config.LLM_BACKEND]
    except KeyError:
        raise RuntimeError(
            f"Unknown LLM_BACKEND '{config.LLM_BACKEND}'. Options: {list(BACKENDS)}"
        )
    return backend(build_prompt(question, chunks))
