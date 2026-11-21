import pytest
from fastapi.testclient import TestClient

from app import llm
from app.main import create_app


class FakeRetriever:
    chunks = [{}] * 3

    def __init__(self, score=0.8):
        self.score = score

    def retrieve(self, question, k):
        return [
            {"doc_id": i, "chunk_id": 0, "source": "IN-Abs",
             "text": "passage " * 100, "score": self.score - 0.1 * i}
            for i in range(k)
        ]


def make_client(monkeypatch, score=0.8, min_score=0.0, answer="The court held X [1]."):
    calls = []

    def fake_generate(question, chunks):
        calls.append(question)
        return answer

    monkeypatch.setattr(llm, "generate_answer", fake_generate)
    app = create_app(lambda: FakeRetriever(score), min_score=min_score)
    return TestClient(app), calls


def test_health(monkeypatch):
    client, _ = make_client(monkeypatch)
    with client:
        r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "chunks": 3}


def test_query_returns_answer_and_numbered_sources(monkeypatch):
    client, _ = make_client(monkeypatch)
    with client:
        r = client.post("/query", json={"question": "What did the court hold?", "top_k": 3})
    body = r.json()
    assert r.status_code == 200
    assert body["found"] is True and body["answer"].startswith("The court held")
    assert [s["ref"] for s in body["sources"]] == [1, 2, 3]
    assert len(body["sources"][0]["snippet"]) <= 300


def test_not_found_answer_sets_flag(monkeypatch):
    client, _ = make_client(monkeypatch, answer=llm.NOT_FOUND)
    with client:
        body = client.post("/query", json={"question": "capital of France?"}).json()
    assert body["found"] is False


def test_low_score_skips_llm(monkeypatch):
    client, calls = make_client(monkeypatch, score=0.1, min_score=0.3)
    with client:
        body = client.post("/query", json={"question": "something unrelated"}).json()
    assert body["found"] is False and body["sources"] == [] and calls == []


@pytest.mark.parametrize("payload", [{"question": "hi"}, {"question": "valid question", "top_k": 0},
                                     {"question": "valid question", "top_k": 99}, {}])
def test_validation_errors(monkeypatch, payload):
    client, _ = make_client(monkeypatch)
    with client:
        assert client.post("/query", json=payload).status_code == 422


def test_llm_failure_returns_502(monkeypatch):
    def boom(question, chunks):
        raise RuntimeError("secret internals")

    monkeypatch.setattr(llm, "generate_answer", boom)
    client = TestClient(create_app(lambda: FakeRetriever()))
    with client:
        r = client.post("/query", json={"question": "valid question"})
    assert r.status_code == 502 and "secret" not in r.text
