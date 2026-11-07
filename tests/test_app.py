from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rag_demo import storage
from rag_demo.models import MetricResult
from rag_demo.pipeline import controlled_run, load_corpus

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_ui_loads_fixture_and_displays_no_fake_scores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert all("agenda" not in expander.label.lower() for expander in app.expander)
    button = next(button for button in app.button if button.label == "Load controlled example")
    button.click().run(timeout=30)
    assert not app.exception
    assert any("CONTROLLED FIXTURE" in item.value for item in app.warning)
    assert any("NOT EVALUATED" in item.value for item in app.warning)
    assert (tmp_path / "session.json").exists()


def test_ui_readiness_failure_is_explicit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import httpx

    from rag_demo.pipeline import OllamaClient

    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path)

    def unavailable(self: OllamaClient) -> list[str]:
        raise httpx.ConnectError("local test service unavailable")

    monkeypatch.setattr(OllamaClient, "models", unavailable)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    next(button for button in app.button if button.label == "Check local Ollama").click().run()
    assert not app.exception
    assert any("Ollama unavailable" in item.value for item in app.error)


@pytest.mark.parametrize("same_judge", [True, False])
def test_replay_comparison_only_pairs_matching_live_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, same_judge: bool
) -> None:
    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path)
    corpus = load_corpus()
    baseline = controlled_run(corpus, "good").model_copy(
        update={
            "mode": "live",
            "configuration": "baseline",
            "generator_model": "synthetic-generator",
            "judge_model": "synthetic-judge",
            "evaluation_seconds": 0,
            "metrics": [
                MetricResult(
                    name="Faithfulness",
                    score=0.2,
                    threshold=0.7,
                    passed=False,
                    reason="Synthetic comparison test.",
                )
            ],
        }
    )
    improved = controlled_run(corpus, "good").model_copy(
        update={
            "mode": "live",
            "configuration": "improved",
            "generator_model": "synthetic-generator",
            "judge_model": "synthetic-judge" if same_judge else "different-judge",
            "evaluation_seconds": 0,
            "metrics": [
                MetricResult(
                    name="Faithfulness",
                    score=0.9,
                    threshold=0.7,
                    passed=True,
                    reason="Synthetic comparison test.",
                )
            ],
        }
    )
    storage.save_runs([baseline, improved], tmp_path / "comparison.json")
    app = AppTest.from_file(str(APP)).run(timeout=30)
    next(button for button in app.button if button.label == "Load saved replay").click().run()
    assert not app.exception
    paired = [table.value for table in app.dataframe if "Delta" in table.value.columns]
    if same_judge:
        assert len(paired) == 1
        assert paired[0]["Delta"].tolist() == [0.7]
        assert paired[0]["Baseline"].tolist() == [0.2]
        assert paired[0]["Improved"].tolist() == [0.9]
    else:
        assert not paired
