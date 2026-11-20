import numpy as np

from app.index import VectorIndex


def _unit(v):
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_search_returns_closest_first(tmp_path):
    vecs = _unit([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]])
    idx = VectorIndex(dim=3)
    idx.add(vecs)
    res = idx.search(_unit([[1, 0, 0]]), k=2)
    assert [i for i, _ in res] == [0, 2]
    assert res[0][1] > res[1][1]


def test_save_and_reload(tmp_path):
    idx = VectorIndex(dim=3)
    idx.add(_unit([[1, 0, 0], [0, 1, 0]]))
    path = str(tmp_path / "t.index")
    idx.save(path)
    assert VectorIndex(path=path).size() == 2


def test_k_larger_than_index_skips_missing():
    idx = VectorIndex(dim=2)
    idx.add(_unit([[1, 0]]))
    assert len(idx.search(_unit([[1, 0]]), k=5)) == 1
