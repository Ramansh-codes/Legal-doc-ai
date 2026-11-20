from app.corpus import select_doc_ids

SRC = ["IN-Abs", "UK-Abs", "IN-Abs", "IN-Abs", "UK-Abs"]


def test_filter_keeps_original_indices():
    assert select_doc_ids(SRC, "IN-Abs") == [0, 2, 3]


def test_max_docs_applies_after_filter():
    assert select_doc_ids(SRC, "IN-Abs", 2) == [0, 2]


def test_no_filter():
    assert select_doc_ids(SRC) == [0, 1, 2, 3, 4]
