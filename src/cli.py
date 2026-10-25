"""Command-line entry points (same pipeline as the Streamlit app).

    python -m src.cli status
    python -m src.cli ask "When was the Eiffel Tower built?"
    python -m src.cli download-nli
    python -m src.cli mutate data/my_claims.jsonl          # {"claim": ..., "evidence": ...}
    python -m src.cli eval-fever data/fever/shared_task_dev.jsonl -n 50
    python -m src.cli eval-ragtruth data/ragtruth/response.jsonl -n 20
    python -m src.cli controlled-test "Tell me about malaria." --example malaria
    python -m src.cli dataset-report                        # statistics of the supplied datasets
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

from src.config import Settings
from src.llm.factory import PROVIDER_NAMES


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_status(settings: Settings, _args) -> int:
    from src.llm.factory import generation_model, make_llm_client
    from src.retrieval.wikipedia import WikipediaRetriever
    from src.verification.nli_verifier import nli_dependencies_available

    client = make_llm_client(settings)
    st = client.status()
    wiki_ok, wiki_msg = WikipediaRetriever(
        settings.wiki_api_url, settings.wiki_user_agent, timeout_s=5, max_retries=1
    ).check_connectivity()
    _print_json({
        "generation": {"provider": settings.llm_provider, "endpoint": client.host,
                       "reachable": st.reachable, "models": st.models[:50],
                       "model": generation_model(settings), "model_available": st.model_available,
                       "error": st.error},
        "wikipedia": {"reachable": wiki_ok, "detail": wiki_msg},
        "nli": {"packages_installed": nli_dependencies_available(), "model": settings.nli_model},
        "verifier_setting": settings.verifier,
    })
    return 0


def cmd_ask(settings: Settings, args) -> int:
    from src.llm.base import LLMError
    from src.pipeline import build_pipeline

    pipeline, choice = build_pipeline(settings)
    for note in choice.notes:
        print(f"NOTE: {note}", file=sys.stderr)
    try:
        answer = pipeline.generate_answer(args.question)
    except LLMError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _print_json(pipeline.analyze(answer).to_dict())
    return 0


def cmd_download_nli(settings: Settings, _args) -> int:
    from src.verification.nli_verifier import NLIVerifier

    print(f"Downloading {settings.nli_model} into {settings.model_cache_dir} (about 0.3 GB)...")
    NLIVerifier(settings.nli_model, cache_dir=settings.model_cache_dir, allow_download=True).load()
    print("Done. Set VERIFIER=nli or auto to use it.")
    return 0


def cmd_mutate(settings: Settings, args) -> int:
    from src.experiments.mutation import (
        LLMMutator,
        MutationValidator,
        ReferenceItem,
        RuleBasedMutator,
        generate_candidates,
        run_experiment,
    )
    from src.pipeline import build_pipeline

    items = []
    for line in Path(args.file).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            items.append(ReferenceItem(row["claim"], row["evidence"], row.get("title", ""), row.get("url", "")))
    pipeline, _ = build_pipeline(settings)
    validator = MutationValidator(None if args.rule_only_validation else pipeline.verifier)
    llm = LLMMutator(pipeline.client, seed=settings.random_seed) if args.llm else None
    candidates, warnings = generate_candidates(
        items, RuleBasedMutator(settings.random_seed), llm, validator
    )
    result = run_experiment(
        items, candidates, pipeline.verify_claim_text,
        {"verifier": pipeline.verifier.method, "llm_mutations": bool(args.llm),
         "validator_same_as_detector": not args.rule_only_validation},
        settings.random_seed,
    )
    result.warnings = warnings + result.warnings
    if result.report:
        print(f"Saved report: {result.report.save(settings.results_dir)}", file=sys.stderr)
    _print_json(result.to_dict())
    return 0


def cmd_eval_fever(settings: Settings, args) -> int:
    from src.dataset_eval import evaluate_fever
    from src.datasets.fever_loader import load_fever_jsonl
    from src.pipeline import build_pipeline

    examples, load_report = load_fever_jsonl(Path(args.file))
    pipeline, _ = build_pipeline(settings)
    run = evaluate_fever(examples, pipeline, args.n, settings.random_seed, load_report.split)
    if run.report:
        print(f"Saved report: {run.report.save(settings.results_dir)}", file=sys.stderr)
    _print_json({"load": asdict(load_report), "report": run.report.to_dict() if run.report else None,
                 "warnings": run.warnings})
    return 0


def cmd_eval_ragtruth(settings: Settings, args) -> int:
    from src.dataset_eval import evaluate_ragtruth
    from src.datasets.ragtruth_loader import load_ragtruth
    from src.pipeline import build_pipeline

    examples, schema = load_ragtruth(Path(args.file), Path(args.source) if args.source else None)
    pipeline, _ = build_pipeline(settings)
    tasks = tuple(args.tasks.split(",")) if args.tasks else None
    run = evaluate_ragtruth(examples, pipeline, args.n, settings.random_seed, args.split or None, tasks,
                            args.include_implicit_true)
    if run.report:
        print(f"Saved report: {run.report.save(settings.results_dir)}", file=sys.stderr)
        for r in run.extra_reports.values():
            r.save(settings.results_dir)
    _print_json({"schema": asdict(schema), "report": run.report.to_dict() if run.report else None,
                 "other": {k: v.to_dict() for k, v in run.extra_reports.items()}, "warnings": run.warnings})
    return 0


def cmd_controlled_test(settings: Settings, args) -> int:
    from src.experiments.injection import insert_claim, load_reviewed_examples, run_controlled_test
    from src.llm.base import LLMError
    from src.pipeline import build_pipeline

    examples = {e.id: e for e in load_reviewed_examples()}
    if args.example not in examples:
        print(f"Unknown example; choose from {sorted(examples)}", file=sys.stderr)
        return 2
    ex = examples[args.example]
    pipeline, _ = build_pipeline(settings)
    try:
        answer = pipeline.generate_answer(args.question)
    except LLMError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    inj = insert_claim(answer, ex.false_claim, ex.transformation, f"reviewed example '{ex.id}'")
    res = run_controlled_test(pipeline, answer, inj)
    _print_json({
        "original_answer (Llama output)": answer.text,
        "test_copy (NOT model output)": inj.modified_answer,
        "injection": asdict(inj),
        "injected_claim_verdicts": [{"claim": c.claim.text, "outcome": c.outcome.value,
                                     "explanation": c.verdict.explanation} for c in res.injected_claims],
        "detected": res.detected,
        "original_summary": asdict(res.original.summary),
        "modified_summary": asdict(res.modified.summary),
    })
    return 0


def cmd_dataset_report(settings: Settings, args) -> int:
    """Statistics of the supplied FEVER/RAGTruth files (no model involved)."""
    from collections import Counter

    from src.datasets.fever_loader import load_fever_jsonl
    from src.datasets.ragtruth_loader import dataset_statistics, load_ragtruth, sentence_labels

    out: dict = {"fever": {}, "ragtruth": None}
    for f in sorted((settings.data_dir / "fever").glob("*.jsonl")):
        exs, rep = load_fever_jsonl(f)
        out["fever"][f.name] = {"rows": rep.n_loaded, "malformed": rep.n_malformed,
                                "unlabelled": rep.n_unlabelled, "unknown_labels": rep.unknown_labels,
                                "classes": dict(Counter(e.label.value if e.label else "unlabelled" for e in exs)),
                                "with_evidence_ids": sum(bool(e.evidence_sets) and e.evidence_sets[0][0].wiki_page
                                                         is not None for e in exs)}
    resp = settings.data_dir / "ragtruth" / "response.jsonl"
    if resp.is_file():
        exs, schema = load_ragtruth(resp)
        sent = Counter()
        for e in exs:
            for s, gold in sentence_labels(e):
                sent[f"{e.split}|{e.task_type}|{'candidate' if s.skip_reason is None else 'skipped'}|"
                     f"{'hallucinated' if gold else 'clean'}"] += 1
        out["ragtruth"] = {"schema": {k: v for k, v in asdict(schema).items() if k != "errors"},
                           "statistics": dataset_statistics(exs), "sentence_labels": dict(sorted(sent.items()))}
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    path = settings.results_dir / "dataset_report.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {path}", file=sys.stderr)
    _print_json(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verifier", choices=["auto", "nli", "llm"])
    parser.add_argument("--extractor", choices=["spacy", "llm"])
    parser.add_argument("--provider", choices=PROVIDER_NAMES, help="generation backend (default LLM_PROVIDER)")
    parser.add_argument("--model", help="generation model (OLLAMA_MODEL for ollama, otherwise LLM_MODEL)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("ask")
    p.add_argument("question")
    sub.add_parser("download-nli")
    p = sub.add_parser("mutate")
    p.add_argument("file", help="JSONL with 'claim' and 'evidence' fields")
    p.add_argument("--llm", action="store_true", help="also generate LLM mutations")
    p.add_argument("--rule-only-validation", action="store_true",
                   help="validate with deterministic checks only (avoids circular validation)")
    p = sub.add_parser("eval-fever")
    p.add_argument("file")
    p.add_argument("-n", type=int, default=50)
    p = sub.add_parser("eval-ragtruth")
    p.add_argument("file", help="response.jsonl")
    p.add_argument("--source", help="source_info.jsonl (default: next to response.jsonl)")
    p.add_argument("--split", default="test", help="split to sample from ('' for all)")
    p.add_argument("--tasks", default="QA,Summary", help="comma-separated task types ('' for all)")
    p.add_argument("--include-implicit-true", action="store_true")
    p.add_argument("-n", type=int, default=20)
    p = sub.add_parser("controlled-test")
    p.add_argument("question")
    p.add_argument("--example", default="malaria", help="id from data/controlled_examples.json")
    sub.add_parser("dataset-report")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    overrides = {k: v for k, v in {"verifier": args.verifier, "extractor": args.extractor,
                                    "llm_provider": args.provider}.items() if v}
    settings = replace(settings, **overrides)
    if args.model:
        field = "ollama_model" if settings.llm_provider == "ollama" else "llm_model"
        settings = replace(settings, **{field: args.model})
    handlers = {"status": cmd_status, "ask": cmd_ask, "download-nli": cmd_download_nli,
                "mutate": cmd_mutate, "eval-fever": cmd_eval_fever, "eval-ragtruth": cmd_eval_ragtruth,
                "controlled-test": cmd_controlled_test, "dataset-report": cmd_dataset_report}
    return handlers[args.cmd](settings, args)


if __name__ == "__main__":
    sys.exit(main())
