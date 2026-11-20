import pytest

from app.chunker import chunk_document, chunk_text


def test_overlap_and_coverage():
    words = [f"w{i}" for i in range(700)]
    chunks = chunk_text(" ".join(words), chunk_size=350, overlap=50)
    assert chunks[0].split()[0] == "w0"
    assert chunks[-1].split()[-1] == "w699"
    # consecutive chunks share exactly `overlap` words
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]


def test_no_duplicate_tail_when_exact_fit():
    chunks = chunk_text(" ".join(["a"] * 350), chunk_size=350, overlap=50)
    assert len(chunks) == 1


def test_empty_text():
    assert chunk_text("") == []


def test_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=10)


def test_chunk_document_metadata():
    out = chunk_document(7, "one two three", chunk_size=2, overlap=1, source="IN")
    assert [c["chunk_id"] for c in out] == [0, 1]
    assert all(c["doc_id"] == 7 and c["source"] == "IN" for c in out)
