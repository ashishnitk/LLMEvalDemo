import csv
import io
import json
import os
import tempfile
from pathlib import Path

from rag_demo.models import Run

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def save_runs(runs: list[Run], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps([run.model_dump(mode="json") for run in runs], indent=2)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


def load_runs(path: Path) -> list[Run]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("A results file must contain a list of runs.")
    return [Run.model_validate(item) for item in payload]


def rows(runs: list[Run]) -> list[dict[str, str | float | bool | None]]:
    records: list[dict[str, str | float | bool | None]] = []
    for run in runs:
        common = {
            "run_id": run.id,
            "case": run.case.id,
            "split": run.case.split,
            "mode": run.mode,
            "configuration": run.configuration,
            "generator": run.generator_model,
            "judge": run.judge_model,
            "generation_seconds": run.generation_seconds,
            "evaluation_seconds": run.evaluation_seconds,
            "corpus_sha256": run.corpus_sha256,
            "retrieval_k": run.retrieval_k,
        }
        if not run.metrics:
            records.append({**common, "metric": "NOT EVALUATED", "score": None, "passed": None})
        for metric in run.metrics:
            records.append(
                {
                    **common,
                    "metric": metric.name,
                    "score": metric.score,
                    "threshold": metric.threshold,
                    "passed": metric.passed,
                    "reason": metric.reason,
                    "error": metric.error,
                }
            )
    return records


def csv_export(runs: list[Run]) -> str:
    records = rows(runs)
    if not records:
        return ""
    stream = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for record in records for key in record))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue()


def gate(runs: list[Run]) -> bool:
    return bool(runs) and all(
        run.metrics
        and all(
            metric.passed is True
            and not metric.error
            and metric.score is not None
            and metric.score >= metric.threshold
            for metric in run.metrics
        )
        for run in runs
    )
