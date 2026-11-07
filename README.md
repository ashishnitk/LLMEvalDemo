# RAG Eval Lab

A 60-minute, local-first workshop for a mixed technical and business audience.
The demo compares actual retrieved evidence, generated answers, and DeepEval judge
assessments using entirely fictional company policies.

For the illustrated architecture, Streamlit walkthrough, Windows setup and
presenter guide, see the [12-page PDF guide](docs/RAG_Eval_Lab_Architecture_and_Setup.pdf).

## Quick start on Windows

Python 3.11+ is required. From this directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://127.0.0.1:8501. In **Workshop**, load a controlled example.
This works without Ollama. Fixtures are hand-authored and **have no evaluation scores**.
They are never presented as actual pipeline output or measured improvements.

Runtime dependencies are pinned in [pyproject.toml](pyproject.toml).
The full tested dependency snapshot is recorded in [requirements-lock.txt](requirements-lock.txt).
For reproducible installation, install that snapshot before the editable project:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

## Enable live RAG and DeepEval

Install [Ollama](https://ollama.com/download/windows), then download a model:

```powershell
ollama pull qwen2.5:7b
ollama list
```

Ollama normally starts its local service after installation. If necessary, run
`ollama serve` in a separate terminal. Use **Check local Ollama** in the sidebar.
The generator and judge names must exactly match names listed by `ollama list`.
The default is a starting point, not a certified judge recommendation. Rehearse
on your hardware; model memory usage and evaluation latency can be substantial.
The same model can fill both roles; an independent stronger judge is preferable
when available. Download all models before an offline presentation.

1. In **Live RAG**, select a case and generate baseline, retrieval-improved and/or
   improved answers.
2. Inspect the exact ranked passages and answer at the bottom of the page.
3. Click **Evaluate this run with local DeepEval**.
4. Read scores **and reasons**. Judge failures are displayed as errors, not scores.
5. In **Results & replay**, compare matched runs and download JSON/CSV.
   Select multiple saved files to replay a CLI baseline/improved comparison.

Generation uses Ollama's local HTTP API. Evaluation explicitly constructs
`OllamaModel` and calls each metric's standalone `measure()`, not the hosted
test-run upload workflow. No Confident AI login or hosted API key is needed.
DeepEval telemetry and Streamlit usage statistics are disabled. Only loopback
HTTP model URLs are accepted; do not configure a local proxy that forwards to a
hosted model if you need strict offline operation. Package/model installation
requires network access; runtime does not require a hosted judge.

## Pipeline and intentional changes

**Retrieval is TF-IDF**, not semantic embeddings. Each short document is one
passage. This avoids an embedding-model download and keeps the workshop focused
on evaluation. It is not representative of a production-scale retriever.

| Configuration | Sources | Generator prompt |
|---|---|---|
| baseline | Current and superseded | Simple policy assistant |
| retrieval-improved | Current only | Same simple prompt |
| improved | Current only | Grounding, exceptions, source citations and abstention |

All configurations use the same TF-IDF settings and chosen `k`. Filtering is
applied before ranking. No reference answer or expected source ID is used for live
retrieval/generation. The retrieval-only configuration isolates source filtering
from the prompt change. Do not promise score increases before measuring them.

The dataset contains 14 cases: 9 development, 5 holdout, including 2 unanswerable
questions. Holdout cases are inspectable for teaching, but should not be used to
tune a claimed held-out result.

## Evaluation

| Metric | Meaning | Reference required |
|---|---|---|
| Answer relevancy | Is the answer relevant to the question? | No |
| Faithfulness | Does the answer align with retrieved evidence? | No |
| Contextual recall | Does retrieved evidence support the reference answer? | Yes |
| Contextual precision | Are useful passages ranked before irrelevant ones? | Yes |
| Policy correctness (G-Eval) | Does the answer satisfy current-policy facts and conditions? | Yes |
| Abstention (G-Eval, unanswerable cases) | Does it acknowledge missing information without inventing facts? | Yes |

For unanswerable cases, contextual recall/precision are intentionally omitted:
there is no supporting reference evidence to retrieve. Faithfulness is omitted
when no passages were supplied. The other applicable metrics still run.
The UI's **expected source-ID coverage** is a deterministic retrieval diagnostic,
not a substitute for contextual recall.

The 0.7 threshold is illustrative and adjustable. Relevancy does not establish
correctness; faithfulness can reward an answer based on outdated evidence.
The additional current-policy reference rubric makes that blind spot visible,
but is still an LLM judgment, not a proof. Audit references and reasons with
humans, repeat representative cases, calibrate thresholds and inspect latency.
Citations are requested, not programmatically verified by these metrics.

## CLI, rehearsals and regression gates

Prepare offline, explicitly unscored fixtures:

```powershell
.\.venv\Scripts\rag-demo.exe --fixtures --output results\fixtures.json
```

Score controlled examples (requires local judge):

```powershell
.\.venv\Scripts\rag-demo.exe --fixtures --evaluate --output results\scored-fixtures.json
```

Compare the same development questions, recording actual outputs:

```powershell
.\.venv\Scripts\rag-demo.exe --evaluate --configuration baseline --case hotel-exception --case hotel-claim --output results\baseline.json
.\.venv\Scripts\rag-demo.exe --evaluate --configuration retrieval-improved --case hotel-exception --case hotel-claim --output results\retrieval-improved.json
.\.venv\Scripts\rag-demo.exe --evaluate --configuration improved --case hotel-exception --case hotel-claim --output results\improved.json
```

Run the full holdout gate:

```powershell
.\.venv\Scripts\rag-demo.exe --evaluate --gate --split holdout --output results\holdout.json
```

Demonstrate a deliberately broken fixture suite failing the gate:

```powershell
.\.venv\Scripts\rag-demo.exe --fixtures --evaluate --gate --output results\broken.json
```

Exit codes: **0** completed (and passed if gating), **1** gate failed,
**2** invalid input, model/metric error or persistence failure.
A metric error can never pass a gate. Unscored runs cannot be gated.
No fixed scores or expected gate outcomes are fabricated: a weak judge may miss
deliberate errors, which is itself worth discussing.

Runs are saved after every completed case, including metric errors. JSON contains
answers, ordered evidence, references, configuration, corpus hash, models,
thresholds, timestamps, and separate generation/evaluation latency. CSV is a
flattened metric export. Replays retain the original mode and timestamps.
Keep CLI output names distinct: the default `latest.json` is overwritten.
The UI writes its history to `results\session.json`.

## Presenter runbook

| Minute | Action |
|---|---|
| 0–5 | Load `unsupported`. Invite a trust vote before revealing the evidence. |
| 5–12 | Show question → retrieved passages → answer. Separate retriever from generator. |
| 12–20 | Explain the four standard metrics and the current-policy rubric. |
| 20–30 | Evaluate 2–3 fixtures live. Read reasons, not only scores. |
| 30–40 | Contrast `retrieval-miss`, `outdated`, and `bad-abstention`. |
| 40–49 | Generate/evaluate the same question under all three configurations. Compare. |
| 49–55 | Show a gate and inspect its failing case or judge error. |
| 55–60 | Discuss calibration, held-out cases, cost/latency and Q&A. |

Before presenting: pin dependencies, download models, check the local service,
rehearse judge quality and timing, and save actual full-suite results.
Run only 2–3 cases live; load saved results for the rest and explicitly call them
replays. The app does not ship with made-up benchmark scores.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy rag_demo app.py
```

Tests cover retrieval/filtering, dataset links, exact generation context,
model errors, DeepEval field wiring with labelled test doubles, persistence,
unscored/error gate behavior, CLI inputs and UI actions.
Mocked judge tests validate plumbing, not real model quality.
The real DeepEval metric classes are also exercised with synthetic structured
Ollama transport responses while socket connections are blocked. This verifies
the installed API integration, not semantic scoring by an actual local model.

## References

- [DeepEval Ollama integration](https://deepeval.com/integrations/models/ollama)
- [Answer relevancy](https://deepeval.com/docs/metrics-answer-relevancy)
- [Faithfulness](https://deepeval.com/docs/metrics-faithfulness)
- [Contextual recall](https://deepeval.com/docs/metrics-contextual-recall)
- [Contextual precision](https://deepeval.com/docs/metrics-contextual-precision)
- [G-Eval](https://deepeval.com/docs/metrics-llm-evals)
