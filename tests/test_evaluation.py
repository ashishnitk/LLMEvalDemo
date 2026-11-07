from pathlib import Path

import pytest

from rag_demo.evaluation import evaluate_run, source_coverage
from rag_demo.models import MetricResult
from rag_demo.pipeline import controlled_run, load_corpus
from rag_demo.storage import csv_export, gate, load_runs, save_runs


def test_source_coverage_is_not_a_judge_score() -> None:
    corpus = load_corpus()
    assert source_coverage(controlled_run(corpus, "good")) == 1
    assert source_coverage(controlled_run(corpus, "retrieval-miss")) == 0.5
    assert source_coverage(controlled_run(corpus, "outdated")) == 0
    assert source_coverage(controlled_run(corpus, "bad-abstention")) is None


def test_gate_requires_scores_and_rejects_errors() -> None:
    run = controlled_run(load_corpus(), "good")
    assert not gate([])
    assert not gate([run])
    passed = MetricResult(name="faithfulness", score=1, threshold=0.7, passed=True)
    failed = passed.model_copy(update={"score": 0.5, "passed": False})
    error = MetricResult(name="faithfulness", threshold=0.7, error="invalid judge JSON")
    assert gate([run.model_copy(update={"metrics": [passed]})])
    assert not gate([run.model_copy(update={"metrics": [failed]})])
    assert not gate([run.model_copy(update={"metrics": [error]})])
    inconsistent = passed.model_copy(update={"score": 0.1})
    assert not gate([run.model_copy(update={"metrics": [inconsistent]})])


def test_round_trip_preserves_evidence_scores_and_errors(tmp_path: Path) -> None:
    run = controlled_run(load_corpus(), "unsupported")
    metric = MetricResult(name="faithfulness", threshold=0.7, error="judge unavailable")
    run = run.model_copy(update={"metrics": [metric]})
    path = tmp_path / "results.json"
    save_runs([run], path)
    assert load_runs(path) == [run]
    assert "judge unavailable" in csv_export([run])
    assert "NOT EVALUATED" in csv_export([controlled_run(load_corpus(), "good")])
    assert not list(tmp_path.glob("*.tmp"))


def test_threshold_checked_before_any_judge_calls() -> None:
    with pytest.raises(ValueError, match="Threshold"):
        evaluate_run(controlled_run(load_corpus(), "good"), threshold=1.1)


def test_deepeval_wiring_and_explicit_metric_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import deepeval.metrics
    import deepeval.models

    measured = []
    judges = []

    class FakeMetric:
        __name__ = "test metric"
        score = 0.9
        reason = "controlled judge double"

        def __init__(self, **kwargs: object) -> None:
            self.__name__ = str(kwargs.get("name", "test metric"))

        def measure(self, case: object) -> None:
            measured.append(case)

        def is_successful(self) -> bool:
            return True

    class BrokenMetric(FakeMetric):
        __name__ = "broken"

        def measure(self, case: object) -> None:
            raise ValueError("bad JSON")

    def fake_judge(**kwargs: object) -> object:
        judges.append(kwargs)
        return object()

    monkeypatch.setattr(deepeval.models, "OllamaModel", fake_judge)
    for name in (
        "AnswerRelevancyMetric",
        "FaithfulnessMetric",
        "ContextualRecallMetric",
        "ContextualPrecisionMetric",
        "GEval",
    ):
        monkeypatch.setattr(deepeval.metrics, name, FakeMetric)
    monkeypatch.setattr(deepeval.metrics, "FaithfulnessMetric", BrokenMetric)
    run = controlled_run(load_corpus(), "good")
    scored = evaluate_run(run, "local-judge", "http://127.0.0.1:11434")
    assert judges == [
        {
            "model": "local-judge",
            "base_url": "http://127.0.0.1:11434",
            "temperature": 0,
            "timeout": 180,
            "trust_env": False,
        }
    ]
    assert len(scored.metrics) == 5
    assert len(measured) == 4
    case = measured[0]
    assert case.retrieval_context == [p.document.text for p in run.passages]
    assert case.expected_output == run.case.expected_answer
    assert scored.metrics[1].error == "ValueError: bad JSON"
    assert scored.metrics[1].score is None
    assert not gate([scored])
    measured.clear()
    unanswerable = evaluate_run(controlled_run(load_corpus(), "bad-abstention"))
    assert len(unanswerable.metrics) == 3
    assert unanswerable.metrics[-1].name == "Abstention"
