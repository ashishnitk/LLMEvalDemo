# RAG Eval Lab: Plain-English Guide and Application Demo

This guide explains the project without assuming machine-learning knowledge.
It covers the four tabs, settings, evaluation results, and a complete
application walkthrough.

## 1. What is this project?

**RAG Eval Lab checks whether an AI assistant gives reliable answers, not just
convincing ones.**

Imagine an assistant that answers employee questions about company policies.
Before employees rely on it, we need to know whether it finds the right
information and follows that information when answering.

The application makes four things visible:

1. The question.
2. The policy evidence available to the assistant.
3. The answer.
4. The evaluation results and reasons.

All company policies in this project are fictional.

### An open-book test analogy

| Project component | Everyday equivalent |
|---|---|
| Policy documents | The textbook |
| Test question | An exam question |
| Retriever | Finding useful pages |
| Generator | Writing an answer from those pages |
| Reference answer | A marking guide |
| DeepEval and the judge model | An automated marking assistant |
| Saved results | The record of the experiment |

The marking assistant can also make mistakes. Its assessments need human review.

### What does RAG mean?

RAG stands for **Retrieval-Augmented Generation**. In everyday language:

> Find information first, then use it to answer.

```text
Question
   |
   v
Find relevant policy passages
   |
   v
Give the question and evidence to the AI model
   |
   v
Generate an answer
   |
   v
Optionally evaluate and save the result
```

This project does not train a new AI model. It uses an existing local model
through Ollama.

## 2. A practical example

**Question:** Can I claim a hotel costing INR 8000 per night?

**Current policy:**

- The standard limit is INR 6000 per night.
- Higher costs require prior written manager approval.
- Without that approval, reimbursement is capped at INR 6000 per night.

**Supported answer:**

> INR 8000 can be reimbursed with prior written manager approval. Without
> approval, reimbursement is capped at INR 6000 per night.

**Incorrect answer:**

> INR 8000 is always reimbursed. No manager approval is needed.

The incorrect answer is relevant to the question, but contradicts the policy.
A confident answer is not necessarily a correct answer.

### Different failures need different fixes

| Failure | What happened? |
|---|---|
| Missing evidence | The assistant did not receive an important policy passage. |
| Unsupported answer | The answer contradicts or goes beyond the supplied evidence. |
| Outdated source | The assistant uses a superseded policy. |
| Off-topic answer | The answer does not address the question. |
| Invented policy | The answer makes up information absent from the documents. |

Finding the right documents and writing the right answer are separate problems.

## 3. What each technology does

| Component | Purpose in this project |
|---|---|
| Python | Implements retrieval, generation, evaluation, and saving results. |
| Streamlit | Provides the website, controls, tables, and run inspector. |
| TF-IDF and cosine similarity | Rank policy documents using weighted word and word-pair matches. |
| Ollama | Runs the local models used to generate and evaluate answers. |
| DeepEval | Organizes the evaluation checks and invokes the judge model. |
| Pydantic | Validates the dataset and result records. |
| JSON and CSV | Preserve complete runs and export tabular results. |

There is no vector database or semantic embedding model in this implementation.
Each short policy document is treated as one passage.

The website and command-line runner use the same Python modules. Generator and
judge are separate roles, even when both use the same model.

## 4. What are the four tabs for?

| Tab | Purpose | Main action |
|---|---|---|
| **Workshop** | Understand prepared examples of success and failure. | Load a controlled example. |
| **Live RAG** | Run the actual assistant on a dataset question. | Generate answers. |
| **Results & replay** | Inspect, compare, export, and reopen experiments. | Review results or load saved replay. |
| **Dataset** | Inspect documents, questions, and expected answers. | Read the underlying test material. |

The tabs are different views of the application. Moving to another tab does not
automatically generate or evaluate anything.

### Workshop: "Show me an example"

Choose a scenario and click **Load controlled example**.

| Scenario | What it demonstrates |
|---|---|
| Correct evidence + supported answer | An answer that includes the essential policy conditions. |
| Missing approval evidence | An incomplete evidence set. |
| Relevant but unsupported answer | An answer that contradicts supplied policy evidence. |
| Answer about meals instead of hotels | An off-topic response. |
| Faithful to a superseded policy | Agreement with outdated evidence rather than current policy. |
| Invented answer to an unanswerable question | Failure to acknowledge missing information. |

These are hand-authored answers and evidence, not naturally observed model
failures. They start without evaluation scores.

Changing the scenario dropdown does not load it automatically. Click
**Load controlled example**, then confirm the selected record in **Inspect a run**.

### Live RAG: "Let the assistant try"

Select a dataset split, question, and one or more configurations. Then click
**Generate answers**.

| Configuration | Eligible sources | Answer instructions |
|---|---|---|
| **baseline** | Current and superseded policies | Simple policy assistant |
| **retrieval-improved** | Current policies only | Same simple instructions |
| **improved** | Current policies only | Follow evidence, include conditions, cite sources, and acknowledge missing information |

The middle configuration isolates the source-filtering change from the prompt
change. The name "improved" describes the intended design, not a guaranteed
score increase.

The live generator receives the question and retrieved passages. It does not
receive the expected answer or expected source IDs.

### Results & replay: "What happened?"

Use this tab to:

- Inspect the current history of runs and metric results.
- Compare matching evaluated live baseline and improved runs.
- Select one or more saved result files and click **Load saved replay**.
- Download CSV or JSON.

Replay loads recorded answers and scores. It does not call the generator or judge.
Loading a replay replaces the in-memory history with the selected saved runs;
download any results you want to retain first.

The comparison requires matching case, corpus, generator, judge, passage count,
and metric threshold. Controlled fixtures and metric errors are excluded.

### Dataset: "What information are we using?"

The dataset contains:

- **9 fictional documents:** 8 current and 1 superseded.
- **14 test cases:** 9 development and 5 holdout.
- Expected answers, expected source IDs, categories, and answerability labels.

Development cases help investigate changes. Holdout cases should be used to
check results without repeatedly tuning the system to those exact questions.

Some questions are intentionally unanswerable from the documents. In those
cases, acknowledging missing information is better than inventing a policy.

## 5. What do the sidebar controls mean?

| Control | Meaning |
|---|---|
| **Ollama URL** | Address of the local model service; normally `http://localhost:11434`. |
| **Generator model** | The installed model that writes answers. |
| **Judge model** | The installed model that evaluates answers through DeepEval. |
| **Retrieved passages (k)** | Maximum number of matching passages supplied to generation; default 3. |
| **Illustrative pass threshold** | Cutoff for metric pass/fail; default 0.7. |
| **Check local Ollama** | Checks reachability, lists installed models, and warns about missing selections. |

More passages do not automatically mean a better answer. Too few can miss
important conditions; too many can introduce irrelevant or conflicting material.

Keep the settings fixed while comparing configurations.

## 6. How to read the shared run inspector

The run inspector appears below the tabs once runs exist.

### Run selection

**Inspect a run** chooses the record being displayed and evaluated.

A label such as:

```text
hotel-exception / unsupported / controlled / example-id
```

identifies the case, configuration or scenario, run mode, and short run ID.
The Workshop scenario selector and the run inspector are separate controls.

### Answer and evidence

The answer column shows:

- The question and response.
- Retrieval-plus-generation time.
- An expandable reference answer and expected source IDs.

The evidence column shows:

- The passages supplied for the run, in order.
- Source IDs, policy versions, and current/superseded labels.
- TF-IDF similarity scores for live runs.

Controlled fixtures use prepared evidence and show `0.00s` generation time.
Their evidence was not selected by a live retrieval operation.

**Expected source-ID coverage** checks how many expected documents were
retrieved. It is not DeepEval contextual recall.

### Status messages

| Message | Meaning |
|---|---|
| **CONTROLLED FIXTURE** | Hand-authored answer and evidence, not live model output. |
| **NOT EVALUATED** | No DeepEval results have been produced for this run. |
| **Evaluation completed and saved locally** | Evaluation finished without recorded metric errors; not necessarily a passing answer. |
| **PASS** | That metric met the configured threshold. |
| **FAIL** | That metric did not meet the threshold. |
| **ERROR** | Evaluation could not produce a valid result for that metric. |

## 7. What does DeepEval check?

Select a run and click **Evaluate this run with local DeepEval**.

The application checks the local judge is installed, builds a test case from the
run, and measures the applicable metrics.

| Metric | Plain-English question |
|---|---|
| **Answer relevancy** | Did the response address the question? |
| **Faithfulness** | Did it agree with the evidence it received? |
| **Contextual recall** | Did the evidence support the facts needed for the reference answer? |
| **Contextual precision** | Were useful passages ranked ahead of irrelevant passages? |
| **Policy correctness** | Did the answer satisfy the current-policy facts and essential conditions? |
| **Abstention** | When information was unavailable, did it acknowledge that without inventing an answer? |

Unanswerable cases use the abstention rubric and omit contextual recall and
precision. Faithfulness is omitted if no passages were supplied.

Important interpretation rules:

- Generating an answer and evaluating it are separate actions.
- A metric can make several model calls, so evaluation may take time.
- Read the **Reason / error** panel, not just the score.
- A score of 0.9 is not a 90% probability of correctness.
- A relevant answer can be wrong.
- An answer faithful to an outdated source can violate current policy.
- A judge can miss an error; its output is not proof.
- Evaluation errors are not low-quality-answer scores and cannot pass a gate.

For the unsupported hotel example, you might expect relevancy and retrieval
checks to look better than faithfulness and correctness. These are expectations,
not guaranteed results. Discuss the actual scores and reasons you receive.

## 8. Start the application

For first-time installation, follow the [project setup instructions](../README.md#quick-start-on-windows).
For an already installed project, run:

```powershell
cd C:\Temp\RAGEvalDemo
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://127.0.0.1:8501`. Keep the terminal open while using the app.
If you explicitly start it with `--server.port 8517`, use port 8517 instead.

For live generation and scoring, install the separate Ollama application and
prepare the selected model:

```powershell
ollama pull qwen2.5:7b
ollama list
Invoke-RestMethod http://localhost:11434/api/tags
```

If Ollama is not serving requests, run `ollama serve` in another terminal.
Do not start a second server if one is already responding.

Use the exact installed model name in the sidebar and click **Check local Ollama**.
Installing the Python `ollama` package alone does not install the model service.

## 9. Complete application demo

Use this order for a complete product walkthrough:

```text
Overview -> Settings -> Dataset -> Workshop -> Run inspector
         -> Live RAG -> Evaluation -> Results and export
```

### Step 1: Introduce the purpose

> This application tests an AI assistant that answers policy questions.
> We can inspect its source information, its answer, and an automated assessment
> of whether it answered properly.

Explain the difference between controlled examples and actual live runs.

### Step 2: Check the settings

Show the generator, judge, passage count, and threshold. Click
**Check local Ollama**.

Start with `k = 3` and threshold `0.7`, and keep them fixed for comparisons.
The website and the model service are separate: displaying a fixture does not
prove that a local model is available.

### Step 3: Inspect the dataset

Open **Dataset**. Show the hotel limit, hotel approval exception, and superseded
hotel policy. Explain why the source version matters.

Briefly show the development and holdout labels and an unanswerable question.

### Step 4: Load a controlled failure

Open **Workshop**, choose **Relevant but unsupported answer**, and click
**Load controlled example**.

Confirm **unsupported / controlled** in the run inspector. Compare the answer
claiming approval is unnecessary with the policy requiring prior approval.

Be explicit that this example was deliberately written to expose a failure mode.

### Step 5: Inspect the complete run

Show the answer, ranked evidence, and reference answer. Point out the
**NOT EVALUATED** message on a newly loaded fixture.

Explain that good evidence does not guarantee a good answer.

### Step 6: Run the actual assistant

Open **Live RAG** and select:

- Split: **development**
- Question: **Can I claim a hotel costing INR 8000 per night?**
- Configurations: **baseline**, **retrieval-improved**, and **improved**

Click **Generate answers**. Use **Inspect a run** to review each output.
Identify which passages were retrieved and whether the answer included the
approval condition.

Do not assume the improved configuration will always win.

### Step 7: Evaluate and interpret

Select a run, click **Evaluate this run with local DeepEval**, and read the results.
Evaluate each live run you want to compare separately.

Start with relevancy, faithfulness, and policy correctness. Then explain the
retrieval metrics. Expand at least one reason.

The green completion message means the checking finished, not that every metric
passed.

### Step 8: Compare and export

Open **Results & replay** and inspect the table and paired baseline/improved
comparison. The retrieval-improved run can also be inspected in the results
table and run inspector; it is not the second column of the paired comparison.

Download:

- **CSV:** a flattened metric table for analysis.
- **JSON:** complete answers, evidence, and evaluation records.

### Step 9: Demonstrate replay

Select saved result files and click **Load saved replay**. Select multiple files
when the baseline and improved runs were saved separately.

Explain that these are recorded results, not fresh model calls.

A fixture file generated with `--fixtures` without `--evaluate` contains
unscored controlled examples. Loading it does not produce scores or populate
the paired live comparison.

### Step 10: Summarize the outcome

> Instead of trusting a convincing answer, we can inspect its evidence,
> identify failures, compare changes, and keep a record of what happened.

The goal is not to make every score green. It is to understand failures and
measure whether a targeted change helped.

## 10. Saving results and keeping the demo reliable

- Check model availability before starting live work.
- Download models before offline use; model downloads can be several GB.
- Prepare actual evaluated results in advance if local judging is slow.
- Clearly identify saved replay instead of presenting it as live evaluation.
- Do not fabricate scores or replace errors with passing results.
- UI history is stored in the browser session and saved to `results\session.json`.
- A new session starts with empty in-memory history; load saved files explicitly.
- Re-evaluating the same run replaces its stored metric results in that history.
  Export first if you need to retain the earlier evaluation.
- The shared session filename is intended for a single-user demo, not isolated
  simultaneous users.
- Local result files are ignored by Git. Another clone will not automatically
  contain your saved runs.

To create unscored fixtures in a fresh clone:

```powershell
.\.venv\Scripts\rag-demo.exe --fixtures --output results\fixtures.json
```

To evaluate those fixtures with a local judge:

```powershell
.\.venv\Scripts\rag-demo.exe --fixtures --evaluate --output results\scored-fixtures.json
```

## 11. Common questions and troubleshooting

| Question or symptom | Explanation or next action |
|---|---|
| The website opens, but evaluation fails with connection refused. | The website and Ollama are separate services. Check `http://localhost:11434/api/tags`, start Ollama, and confirm the sidebar URL. |
| The model is missing. | Run `ollama list`, pull the required model, and use its exact name/tag. |
| The wrong example is displayed. | Click **Load controlled example**, then check **Inspect a run**. |
| Generation time is `0.00s`. | Expected for a hand-authored controlled fixture. Evaluation has a separate duration. |
| Evaluation completed, but some metrics failed. | The evaluation ran successfully and found quality problems. Read the reasons. |
| There are no paired comparison rows. | Evaluate matching live baseline and improved runs, or load both saved files. Check matching settings. |
| The judge returns invalid output or times out. | Inspect the recorded error and local model behavior. Test a compatible judge and suitable hardware; do not treat the error as a valid score. |
| A policy-faithful answer is still wrong. | The source may be outdated or incomplete. Inspect current-policy correctness as well. |
| Can this prove the assistant is production-ready? | No. This small synthetic dataset illustrates evaluation methods, not broad real-world accuracy. |

For architecture diagrams and detailed setup, see the
[student PDF guide](RAG_Eval_Lab_Architecture_and_Setup.pdf).
