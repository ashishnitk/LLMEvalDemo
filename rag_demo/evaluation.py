import os
from time import perf_counter

from rag_demo.models import MetricResult, Run
from rag_demo.pipeline import DEFAULT_MODEL, DEFAULT_URL, local_url

# Disable DeepEval telemetry before its first import. Standalone measure() avoids run uploads.
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"


def source_coverage(run: Run) -> float | None:
    expected = set(run.case.expected_source_ids)
    if not expected:
        return None
    retrieved = {p.document.id for p in run.passages}
    return len(expected & retrieved) / len(expected)


def evaluate_run(
    run: Run,
    judge_model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_URL,
    threshold: float = 0.7,
) -> Run:
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    url = local_url(base_url)
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.metrics.base_metric import BaseMetric
    from deepeval.models import OllamaModel
    from deepeval.test_case import LLMTestCase, SingleTurnParams

    judge = OllamaModel(
        model=judge_model, base_url=url, temperature=0, timeout=180, trust_env=False
    )
    case = LLMTestCase(
        input=run.case.question,
        actual_output=run.answer,
        expected_output=run.case.expected_answer,
        retrieval_context=[p.document.text for p in run.passages],
    )
    metrics: list[BaseMetric] = [
        AnswerRelevancyMetric(threshold=threshold, model=judge, async_mode=False)
    ]
    if run.passages:
        metrics.append(FaithfulnessMetric(threshold=threshold, model=judge, async_mode=False))
    if run.case.answerable:
        metrics.extend(
            [
                ContextualRecallMetric(threshold=threshold, model=judge, async_mode=False),
                ContextualPrecisionMetric(threshold=threshold, model=judge, async_mode=False),
            ]
        )
    metrics.append(
        GEval(
            name="Policy correctness" if run.case.answerable else "Abstention",
            criteria=(
                "Compare the actual output with the expected output as the authoritative current"
                " policy. Require all relevant numeric limits, conditions and approvals."
                " Penalize contradictions, missing essential conditions and invented facts."
                " For a question whose expected output says information is unavailable, the"
                " actual output must explicitly acknowledge the missing information and must"
                " not invent an answer. Paraphrases are acceptable."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            threshold=threshold,
            model=judge,
            async_mode=False,
        )
    )
    start = perf_counter()
    results = []
    for metric in metrics:
        metric_start = perf_counter()
        try:
            metric.measure(case)
            if metric.score is None:
                raise ValueError("Judge did not return a score.")
            results.append(
                MetricResult(
                    name=metric.__name__,
                    score=metric.score,
                    threshold=threshold,
                    passed=metric.is_successful(),
                    reason=metric.reason,
                    duration_seconds=perf_counter() - metric_start,
                )
            )
        except Exception as exc:
            # Preserve per-metric failures as errors, never as zero scores or passing results.
            results.append(
                MetricResult(
                    name=metric.__name__,
                    threshold=threshold,
                    error=f"{type(exc).__name__}: {exc}",
                    duration_seconds=perf_counter() - metric_start,
                )
            )
    return run.model_copy(
        update={
            "metrics": results,
            "judge_model": judge_model,
            "evaluation_seconds": perf_counter() - start,
        }
    )
