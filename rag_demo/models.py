import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Configuration = Literal["baseline", "retrieval-improved", "improved"]
Scenario = Literal[
    "good", "retrieval-miss", "unsupported", "off-topic", "outdated", "bad-abstention"
]
Mode = Literal["live", "controlled"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Document(Record):
    id: str
    title: str
    version: str
    current: bool
    text: str


class Case(Record):
    id: str
    question: str
    expected_answer: str
    expected_source_ids: list[str]
    category: str
    split: Literal["development", "holdout"]
    answerable: bool = True


class Corpus(Record):
    documents: list[Document]
    cases: list[Case]

    @model_validator(mode="after")
    def validate_links(self) -> "Corpus":
        ids = [doc.id for doc in self.documents]
        case_ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)) or len(case_ids) != len(set(case_ids)):
            raise ValueError("Document and case IDs must be unique.")
        current_ids = {doc.id for doc in self.documents if doc.current}
        for case in self.cases:
            if not set(case.expected_source_ids) <= current_ids:
                raise ValueError(f"{case.id}: expected sources must reference current documents.")
            if case.answerable != bool(case.expected_source_ids):
                raise ValueError(f"{case.id}: answerability and expected sources disagree.")
        return self


class Passage(Record):
    document: Document
    score: float


class MetricResult(Record):
    name: str
    score: float | None = None
    threshold: float
    passed: bool | None = None
    reason: str | None = None
    error: str | None = None
    duration_seconds: float = 0

    @model_validator(mode="after")
    def validate_result(self) -> "MetricResult":
        if self.error:
            if self.score is not None or self.passed is not None:
                raise ValueError("Metric errors must not carry scores or pass/fail results.")
        elif self.score is None or self.passed is None:
            raise ValueError("A measured metric requires a score and a pass/fail result.")
        if self.score is not None and (not math.isfinite(self.score) or not 0 <= self.score <= 1):
            raise ValueError("Metric scores must be finite and between 0 and 1.")
        if not 0 <= self.threshold <= 1:
            raise ValueError("Metric thresholds must be between 0 and 1.")
        return self


class Run(Record):
    id: str
    created_at: str
    case: Case
    configuration: str
    mode: Mode
    answer: str
    passages: list[Passage]
    generator_model: str
    generation_seconds: float
    corpus_sha256: str
    metrics: list[MetricResult] = Field(default_factory=list)
    judge_model: str | None = None
    evaluation_seconds: float | None = None
    retrieval_k: int
    schema_version: Literal[1] = 1
