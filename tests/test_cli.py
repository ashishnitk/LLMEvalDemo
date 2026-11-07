import sys
from pathlib import Path

import pytest

from rag_demo.cli import main
from rag_demo.models import MetricResult, Run
from rag_demo.pipeline import DEFAULT_MODEL, DEFAULT_URL, OllamaClient
from rag_demo.storage import load_runs


def test_offline_cli_persists_unscored_fixtures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "fixtures.json"
    monkeypatch.setattr(sys, "argv", ["rag-demo", "--fixtures", "--output", str(path)])
    assert main() == 0
    runs = load_runs(path)
    assert len(runs) == 6
    assert all(run.mode == "controlled" and not run.metrics for run in runs)
    assert path.with_suffix(".csv").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--fixtures", "--gate"],
        ["--case", "missing"],
        ["--k", "0"],
        ["--fixtures", "--case", "hotel-exception"],
        ["--threshold", "1.5"],
    ],
)
def test_cli_rejects_invalid_inputs(monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["rag-demo", *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


@pytest.mark.parametrize("metric_error, expected_exit", [(False, 1), (True, 2)])
def test_cli_gate_failures_are_persistent_and_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, metric_error: bool, expected_exit: int
) -> None:
    import rag_demo.cli

    def installed(self: OllamaClient) -> list[str]:
        return [DEFAULT_MODEL]

    def synthetic_evaluation(
        run: Run,
        judge_model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        threshold: float = 0.7,
    ) -> Run:
        metric = (
            MetricResult(name="test double", threshold=threshold, error="synthetic judge error")
            if metric_error
            else MetricResult(
                name="test double",
                score=0.1,
                passed=False,
                threshold=threshold,
                reason="Synthetic failing score.",
            )
        )
        return run.model_copy(update={"metrics": [metric], "judge_model": judge_model})

    path = tmp_path / "gate.json"
    monkeypatch.setattr(OllamaClient, "models", installed)
    monkeypatch.setattr(rag_demo.cli, "evaluate_run", synthetic_evaluation)
    monkeypatch.setattr(
        sys, "argv", ["rag-demo", "--fixtures", "--evaluate", "--gate", "--output", str(path)]
    )
    assert main() == expected_exit
    assert len(load_runs(path)) == 6
    assert path.with_suffix(".csv").exists()
