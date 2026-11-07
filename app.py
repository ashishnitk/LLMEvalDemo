import json

import httpx
import streamlit as st
from pydantic import ValidationError

from rag_demo.evaluation import evaluate_run, source_coverage
from rag_demo.models import Run
from rag_demo.pipeline import (
    CONFIGURATIONS,
    DEFAULT_MODEL,
    DEFAULT_URL,
    SCENARIOS,
    OllamaClient,
    controlled_run,
    corpus_hash,
    generate,
    load_corpus,
)
from rag_demo.storage import RESULTS_DIR, csv_export, load_runs, rows, save_runs

st.set_page_config(page_title="RAG Eval Lab", page_icon="🔎", layout="wide")

METRIC_GUIDE = [
    {"Metric": "Answer relevancy", "Question": "Is the answer on topic?", "Reference needed": "No"},
    {
        "Metric": "Faithfulness",
        "Question": "Does the answer align with its retrieved evidence?",
        "Reference needed": "No",
    },
    {
        "Metric": "Contextual recall",
        "Question": "Does the evidence support the reference answer?",
        "Reference needed": "Yes",
    },
    {
        "Metric": "Contextual precision",
        "Question": "Are useful passages ranked before irrelevant ones?",
        "Reference needed": "Yes",
    },
    {
        "Metric": "Policy correctness / Abstention (G-Eval)",
        "Question": "Does the answer satisfy our current-policy reference and rubric?",
        "Reference needed": "Yes",
    },
]


def remember(run: Run) -> None:
    previous: list[Run] = st.session_state.history
    history = [item for item in previous if item.id != run.id] + [run]
    save_runs(history, RESULTS_DIR / "session.json")
    st.session_state.history = history
    st.session_state.active = run


def render_run(run: Run) -> None:
    if run.mode == "controlled":
        st.warning("CONTROLLED FIXTURE: hand-authored answer and evidence, not live RAG output.")
    st.caption(
        f"Case: {run.case.id} | {run.case.split} | Configuration: {run.configuration} | "
        f"Generator: {run.generator_model} | Run: {run.id[:8]}"
    )
    st.subheader(run.case.question)
    answer_column, evidence_column = st.columns([1, 1])
    with answer_column:
        st.markdown("#### Answer")
        st.info(run.answer)
        st.metric("Retrieval + generation time", f"{run.generation_seconds:.2f}s")
        with st.expander("Human-reviewed reference answer"):
            st.write(run.case.expected_answer)
            st.write("Expected sources:", run.case.expected_source_ids)
    with evidence_column:
        st.markdown("#### Evidence actually supplied to generation")
        if not run.passages:
            st.warning("No passages retrieved.")
        for rank, passage in enumerate(run.passages, start=1):
            doc = passage.document
            status = "CURRENT" if doc.current else "SUPERSEDED"
            with st.expander(
                f"{rank}. {doc.id} | {status} {doc.version} | "
                f"{'fixture rank' if run.mode == 'controlled' else f'TF-IDF {passage.score:.3f}'}",
                expanded=True,
            ):
                st.write(doc.text)
        coverage = source_coverage(run)
        if coverage is not None:
            st.caption(
                f"Expected source-ID coverage: {coverage:.0%}. "
                "Deterministic diagnostic, NOT DeepEval contextual recall."
            )
    if not run.metrics:
        st.warning("NOT EVALUATED. No DeepEval scores have been generated for this run.")
        return
    st.markdown("#### DeepEval results")
    st.caption(
        f"Local judge: {run.judge_model} | Evaluation time: {run.evaluation_seconds:.2f}s. "
        "Thresholds are illustrative; scores are judge assessments, not proof."
    )
    columns = st.columns(len(run.metrics))
    for column, metric in zip(columns, run.metrics, strict=True):
        with column:
            if metric.error:
                st.error(f"{metric.name}: ERROR")
            else:
                st.metric(metric.name, f"{metric.score:.2f}")
                st.write("PASS" if metric.passed else "FAIL")
            with st.expander("Reason / error"):
                st.write(metric.error or metric.reason or "No reason returned by the judge.")
                st.caption(f"Threshold {metric.threshold:.2f}; {metric.duration_seconds:.2f}s")


def comparison(runs: list[Run]) -> None:
    live = [run for run in runs if run.mode == "live" and run.metrics]
    latest: dict[tuple[str, str, str, str, str | None, int], Run] = {}
    for run in live:
        key = (
            run.case.id,
            run.configuration,
            run.corpus_sha256,
            run.generator_model,
            run.judge_model,
            run.retrieval_k,
        )
        latest[key] = run
    pairs = []
    for key, improved in latest.items():
        if improved.configuration != "improved":
            continue
        baseline_key = (key[0], "baseline", *key[2:])
        baseline = latest.get(baseline_key)
        if baseline is None:
            continue
        before = {metric.name: metric for metric in baseline.metrics}
        for after in improved.metrics:
            prior = before.get(after.name)
            if (
                prior is None
                or prior.error
                or after.error
                or prior.threshold != after.threshold
                or prior.score is None
                or after.score is None
            ):
                continue
            pairs.append(
                {
                    "Case": improved.case.id,
                    "Split": improved.case.split,
                    "Metric": after.name,
                    "Baseline": prior.score,
                    "Improved": after.score,
                    "Delta": round(after.score - prior.score, 4),
                    "Threshold": after.threshold,
                    "Generator": improved.generator_model,
                    "Judge": improved.judge_model,
                    "k": improved.retrieval_k,
                }
            )
    if pairs:
        st.dataframe(pairs, hide_index=True, width="stretch")
        st.caption(
            "Latest matching live runs only: same case, corpus, generator, judge, k and threshold. "
            "Controlled fixtures and metric errors are excluded; no causal or statistical claim."
        )
    else:
        st.info("Evaluate matching baseline and improved live runs to see measured deltas.")


def main() -> None:
    corpus = load_corpus()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "active" not in st.session_state:
        st.session_state.active = None
    st.title("RAG Eval Lab")
    st.markdown("### Your answer sounds right. Can you prove it?")
    st.caption("Synthetic company policies | Local Python + Ollama + DeepEval")
    with st.sidebar:
        st.header("Local model settings")
        url = st.text_input("Ollama URL", DEFAULT_URL)
        generator_model = st.text_input("Generator model", DEFAULT_MODEL)
        judge_model = st.text_input("Judge model", DEFAULT_MODEL)
        k = st.slider("Retrieved passages (k)", 1, len(corpus.documents), 3)
        threshold = st.slider("Illustrative pass threshold", 0.0, 1.0, 0.7, 0.05)
        st.caption("A judge may make multiple model calls per metric. Evaluation can take time.")
        if st.button("Check local Ollama"):
            try:
                installed = OllamaClient(url, generator_model).models()
                st.success("Ollama is responding.")
                st.write("Installed models:", installed)
                missing = {generator_model, judge_model} - set(installed)
                if missing:
                    st.warning(f"Models not installed: {sorted(missing)}")
            except (httpx.HTTPError, ValueError) as exc:
                st.error(f"Ollama unavailable: {exc}. Install Ollama and pull the selected models.")
        st.divider()
        st.caption("No fabricated scores. No hosted judge. No Confident AI login required.")

    workshop, live_tab, results_tab, dataset_tab = st.tabs(
        ["1 · Workshop", "2 · Live RAG", "3 · Results & replay", "4 · Dataset"]
    )
    with workshop:
        st.subheader("Start with the failure, not the framework")
        st.dataframe(METRIC_GUIDE, hide_index=True, width="stretch")
        st.warning(
            "Relevant does not mean correct. Faithful to an outdated source does not mean "
            "correct under current policy. Use the reference-based policy rubric too."
        )
        scenario = st.selectbox(
            "Controlled scenario", list(SCENARIOS), format_func=lambda name: SCENARIOS[name]
        )
        if st.button("Load controlled example", type="primary"):
            try:
                remember(controlled_run(corpus, scenario))
            except OSError as exc:
                st.error(f"Could not persist the example: {exc}")
        st.caption(
            "These fixtures isolate failure modes. They do not simulate natural model errors."
        )
    with live_tab:
        st.subheader("Run the actual pipeline")
        st.write(
            "Baseline includes all policy versions and a simple prompt. Retrieval-improved filters "
            "to current sources only. Improved keeps that filter and adds grounding, citations "
            "and abstention instructions. All use the same TF-IDF ranking and chosen k."
        )
        split = st.selectbox("Dataset split", ["development", "holdout", "all"])
        available_cases = [case for case in corpus.cases if split == "all" or case.split == split]
        with st.form("live_run"):
            st.caption("Holdout cases are for evaluation, not tuning.")
            case_id = st.selectbox(
                "Question",
                [case.id for case in available_cases],
                format_func=lambda cid: next(c.question for c in corpus.cases if c.id == cid),
            )
            configs = st.multiselect(
                "Configurations", CONFIGURATIONS, default=["baseline", "improved"]
            )
            submitted = st.form_submit_button("Generate answers", type="primary")
        if submitted:
            selected = next(case for case in corpus.cases if case.id == case_id)
            if not configs:
                st.error("Select at least one configuration.")
            else:
                for configuration in configs:
                    try:
                        with st.spinner(f"Generating {configuration} locally..."):
                            remember(
                                generate(
                                    corpus,
                                    selected,
                                    configuration,
                                    OllamaClient(url, generator_model),
                                    k,
                                )
                            )
                    except (httpx.HTTPError, ValueError, OSError) as exc:
                        st.error(f"{configuration} failed: {exc}")
                        break

    with results_tab:
        st.subheader("Measured results, not promised improvements")
        saved = sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.exists() else []
        if saved:
            selected_files = st.multiselect(
                "Saved local results (select multiple to compare)",
                saved,
                default=[saved[0]],
                format_func=lambda p: p.name,
            )
            if st.button("Load saved replay"):
                try:
                    if not selected_files:
                        raise ValueError("Select at least one saved results file.")
                    by_id = {run.id: run for path in selected_files for run in load_runs(path)}
                    history = sorted(by_id.values(), key=lambda run: run.created_at)
                    st.session_state.history = history
                    st.session_state.active = history[-1] if history else None
                    st.success("Loaded saved results. This is a replay, not a fresh evaluation.")
                except (OSError, ValueError, ValidationError) as exc:
                    st.error(f"Could not load saved results: {exc}")
        history = st.session_state.history
        if history:
            st.dataframe(rows(history), hide_index=True, width="stretch")
            st.markdown("#### Paired baseline → improved")
            comparison(history)
            st.download_button(
                "Download CSV", csv_export(history), "rag-eval-results.csv", "text/csv"
            )
            st.download_button(
                "Download JSON (answers, evidence and scores)",
                json.dumps([run.model_dump(mode="json") for run in history], indent=2),
                "rag-eval-results.json",
                "application/json",
            )
        else:
            st.info("Load a controlled example or run a live question to begin.")

    with dataset_tab:
        st.subheader("Small, inspectable, human-reviewed test set")
        st.write("14 cases: 9 development and 5 holdout. The data is entirely fictional.")
        st.dataframe([case.model_dump() for case in corpus.cases], hide_index=True, width="stretch")
        for doc in corpus.documents:
            with st.expander(
                f"{doc.title} | {doc.version} | {'CURRENT' if doc.current else 'SUPERSEDED'}"
            ):
                st.write(doc.text)
        st.caption(f"Corpus fingerprint: {corpus_hash(corpus)}")

    st.divider()
    history = st.session_state.history
    if history:
        selected_run_id = st.selectbox(
            "Inspect a run",
            [run.id for run in history],
            index=len(history) - 1,
            format_func=lambda rid: next(
                f"{r.case.id} / {r.configuration} / {r.mode} / {r.id[:8]}"
                for r in history
                if r.id == rid
            ),
        )
        active = next(run for run in history if run.id == selected_run_id)
        if active.corpus_sha256 != corpus_hash(corpus):
            st.warning("This saved run uses a different corpus version than the current dataset.")
        if st.button("Evaluate this run with local DeepEval", type="primary"):
            try:
                installed = OllamaClient(url, judge_model).models()
                if judge_model not in installed:
                    raise ValueError(
                        f"Judge {judge_model} is missing. Run ollama pull {judge_model}."
                    )
                with st.spinner("Evaluating locally. Each metric may make several judge calls..."):
                    active = evaluate_run(active, judge_model, url, threshold)
                    remember(active)
                if any(metric.error for metric in active.metrics):
                    st.error(
                        "Evaluation has metric errors. Inspect details; "
                        "this run cannot pass a gate."
                    )
                else:
                    st.success("Evaluation completed and saved locally.")
            except (httpx.HTTPError, ValueError, ImportError, OSError) as exc:
                st.error(f"Evaluation failed: {exc}")
        render_run(active)


main()
