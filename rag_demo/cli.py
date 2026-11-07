import argparse
import sys
from pathlib import Path

import httpx

from rag_demo.evaluation import evaluate_run
from rag_demo.pipeline import (
    CONFIGURATIONS,
    DEFAULT_MODEL,
    DEFAULT_URL,
    SCENARIOS,
    OllamaClient,
    controlled_run,
    generate,
    load_corpus,
)
from rag_demo.storage import RESULTS_DIR, csv_export, gate, save_runs


def main() -> int:
    parser = argparse.ArgumentParser(description="Local RAG evaluation and regression gate")
    parser.add_argument("--fixtures", action="store_true", help="Use labelled controlled examples")
    parser.add_argument("--evaluate", action="store_true", help="Score with a local DeepEval judge")
    parser.add_argument("--gate", action="store_true", help="Exit 1 for any failure or judge error")
    parser.add_argument("--case", action="append", help="Case ID, repeat to select multiple cases")
    parser.add_argument("--split", choices=["development", "holdout", "all"], default="development")
    parser.add_argument("--configuration", choices=CONFIGURATIONS, default="improved")
    parser.add_argument("--generator", default=DEFAULT_MODEL)
    parser.add_argument("--judge", default=DEFAULT_MODEL)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "latest.json")
    args = parser.parse_args()
    if args.gate and not args.evaluate:
        parser.error("--gate requires --evaluate; unscored runs cannot pass a gate.")
    if args.fixtures and args.case:
        parser.error("--case applies to live runs, not the fixed controlled scenarios.")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")
    corpus = load_corpus()
    if not 1 <= args.k <= len(corpus.documents):
        parser.error(f"--k must be between 1 and {len(corpus.documents)}.")
    requested = set(args.case or [])
    known = {case.id for case in corpus.cases}
    if requested - known:
        parser.error(f"Unknown case IDs: {sorted(requested - known)}")
    cases = [
        case
        for case in corpus.cases
        if (case.id in requested if requested else args.split == "all" or case.split == args.split)
    ]
    runs = []
    try:
        client = OllamaClient(args.url, args.generator)
        if args.evaluate or not args.fixtures:
            installed = client.models()
            needed = {args.judge} if args.evaluate else set()
            if not args.fixtures:
                needed.add(args.generator)
            missing = needed - set(installed)
            if missing:
                raise ValueError(f"Missing Ollama models: {sorted(missing)}. Run ollama pull.")
        work = list(SCENARIOS) if args.fixtures else cases
        for item in work:
            run = (
                controlled_run(corpus, item)
                if isinstance(item, str)
                else generate(corpus, item, args.configuration, client, args.k)
            )
            if args.evaluate:
                run = evaluate_run(run, args.judge, args.url, args.threshold)
            runs.append(run)
            save_runs(runs, args.output)
            print(f"{run.case.id} / {run.configuration}:")
            print(run.answer)
            for metric in run.metrics:
                status = "ERROR" if metric.error else "PASS" if metric.passed else "FAIL"
                print(f"  {metric.name}: {status} score={metric.score}")
                print(f"    {metric.error or metric.reason}")
        args.output.with_suffix(".csv").write_text(csv_export(runs), encoding="utf-8")
        print(f"Saved {len(runs)} runs to {args.output}")
        if any(metric.error for run in runs for metric in run.metrics):
            return 2
        return 1 if args.gate and not gate(runs) else 0
    except (httpx.HTTPError, ValueError, ImportError, OSError) as exc:
        print(f"Demo failed: {exc}", file=sys.stderr)
        if runs:
            print(f"Completed runs were saved to {args.output}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
