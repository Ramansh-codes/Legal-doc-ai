from app.llm import NOT_FOUND, build_prompt


def test_prompt_numbers_passages_and_requires_citations():
    p = build_prompt("Who won?", [{"text": "alpha"}, {"text": "beta"}])
    assert "[1] alpha" in p and "[2] beta" in p
    assert "cite" in p.lower()
    assert NOT_FOUND in p
