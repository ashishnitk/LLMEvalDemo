from collections.abc import Callable

import httpx
import pytest

from rag_demo.models import Corpus
from rag_demo.pipeline import (
    SCENARIOS,
    OllamaClient,
    controlled_run,
    generate,
    load_corpus,
    local_url,
    retrieve,
)


def test_dataset_is_consistent() -> None:
    corpus = load_corpus()
    assert len(corpus.cases) == 14
    assert sum(case.split == "holdout" for case in corpus.cases) == 5
    assert sum(not case.answerable for case in corpus.cases) == 2
    assert len({doc.id for doc in corpus.documents}) == len(corpus.documents)


def test_invalid_dataset_links_rejected() -> None:
    raw = load_corpus().model_dump()
    raw["cases"][0]["expected_source_ids"] = ["missing"]
    with pytest.raises(ValueError, match="current documents"):
        Corpus.model_validate(raw)


def test_retrieval_filter_and_ranking() -> None:
    corpus = load_corpus()
    question = "Can I claim a hotel costing INR 8000 per night?"
    baseline = retrieve(corpus, question, "baseline", 3)
    improved = retrieve(corpus, question, "improved", 3)
    assert any(p.document.id == "hotel-old" for p in baseline)
    assert all(p.document.current for p in improved)
    assert {"hotel-limit", "hotel-approval"} <= {p.document.id for p in improved}
    assert [p.score for p in improved] == sorted([p.score for p in improved], reverse=True)
    assert retrieve(corpus, question, "retrieval-improved", 3) == improved


def test_no_overlap_does_not_return_arbitrary_passages() -> None:
    assert retrieve(load_corpus(), "xyzabc", "improved") == []


@pytest.mark.parametrize("k", [0, -1, 100])
def test_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must"):
        retrieve(load_corpus(), "hotel", "improved", k)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://example.com",
        "http://localhost.evil.com",
        "http://user:pass@localhost:11434",
        "http://localhost:11434/api",
        "http://localhost:11434/?redirect=example.com",
    ],
)
def test_only_local_model_endpoints(url: str) -> None:
    with pytest.raises(ValueError, match="local HTTP"):
        local_url(url)


def test_controlled_scenarios_are_not_live_runs() -> None:
    corpus = load_corpus()
    for scenario in SCENARIOS:
        run = controlled_run(corpus, scenario)
        assert run.mode == "controlled"
        assert run.generation_seconds == 0
        assert run.metrics == []
        assert run.generator_model.startswith("hand-authored")
    assert controlled_run(corpus, "retrieval-miss").passages[0].document.id == "hotel-limit"
    assert not controlled_run(corpus, "outdated").passages[0].document.current
    assert not controlled_run(corpus, "bad-abstention").case.answerable


def mock_http(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )


def test_generation_sends_exact_retrieved_context(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "INR 6000. [hotel-limit]"}})

    mock_http(monkeypatch, handler)
    corpus = load_corpus()
    case = corpus.cases[0]
    run = generate(corpus, case, "improved", OllamaClient(), 3)
    assert run.mode == "live"
    assert run.answer == "INR 6000. [hotel-limit]"
    for passage in run.passages:
        assert passage.document.text in requests[0]["messages"][1]["content"]
    assert "never invent" in requests[0]["messages"][0]["content"]
    assert "SUPERSEDED" not in requests[0]["messages"][1]["content"]


def test_model_failure_is_not_a_fake_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_http(monkeypatch, lambda request: httpx.Response(404, json={"error": "missing model"}))
    with pytest.raises(httpx.HTTPStatusError):
        OllamaClient().answer("question", [], True)


def test_empty_model_output_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_http(monkeypatch, lambda request: httpx.Response(200, json={"message": {"content": ""}}))
    with pytest.raises(ValueError, match="empty answer"):
        OllamaClient().answer("question", [], True)
