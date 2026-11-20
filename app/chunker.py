def chunk_text(text, chunk_size=350, overlap=50):
    """Split text into overlapping word windows."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        if i + chunk_size >= len(words):  # last window reached the end
            break
        i += step

    return chunks


def chunk_document(doc_id, text, chunk_size=350, overlap=50, source=""):
    """Chunk one document and attach metadata so answers can cite sources."""
    return [
        {"doc_id": doc_id, "chunk_id": n, "source": source, "text": c}
        for n, c in enumerate(chunk_text(text, chunk_size, overlap))
    ]
