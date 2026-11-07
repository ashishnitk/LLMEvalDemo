import json
import socket

import ollama
import pytest

from rag_demo.evaluation import evaluate_run
from rag_demo.pipeline import controlled_run, load_corpus
from rag_demo.storage import gate


def test_actual_deepeval_metrics_use_ollama_structured_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real metrics with synthetic transport responses, not a real judge."""
    requested_schemas = []

    def no_network(self: socket.socket, address: object) -> None:
        raise AssertionError(f"Unexpected network connection: {address}")

    def synthetic_chat(self: ollama.Client, **kwargs: object) -> ollama.ChatResponse:
        schema = kwargs["format"]
        assert isinstance(schema, dict)
        requested_schemas.append(schema["title"])
        properties = schema["properties"]
        if "statements" in properties:
            payload = {"statements": ["The standard hotel limit is INR 6000 per night."]}
        elif "truths" in properties:
            payload = {"truths": ["The standard hotel limit is INR 6000 per night."]}
        elif "claims" in properties:
            payload = {"claims": ["The standard hotel limit is INR 6000 per night."]}
        elif "verdicts" in properties:
            payload = {
                "verdicts": [
                    {"verdict": "yes", "reason": "Synthetic transport verdict."},
                    {"verdict": "yes", "reason": "Synthetic transport verdict."},
                ]
            }
        elif "steps" in properties:
            payload = {"steps": ["Compare policy limits.", "Compare approval requirements."]}
        elif "score" in properties:
            payload = {"score": 9, "reason": "Synthetic transport score, not a real judgment."}
        else:
            payload = {"reason": "Synthetic transport reason."}
        return ollama.ChatResponse(
            model="qwen2.5:7b",
            message=ollama.Message(role="assistant", content=json.dumps(payload)),
            done=True,
        )

    monkeypatch.setattr(ollama.Client, "chat", synthetic_chat)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    run = evaluate_run(controlled_run(load_corpus(), "good"))
    assert all(metric.error is None for metric in run.metrics), [
        metric.error for metric in run.metrics
    ]
    assert [metric.name for metric in run.metrics] == [
        "Answer Relevancy",
        "Faithfulness",
        "Contextual Recall",
        "Contextual Precision",
        "Policy correctness [GEval]",
    ]
    assert all(metric.score is not None for metric in run.metrics)
    assert gate([run])
    assert {"Statements", "Truths", "Claims", "Verdicts", "Steps", "ReasonScore"} <= set(
        requested_schemas
    )
