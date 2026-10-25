"""Run the detector on labelled datasets and score it with :mod:`src.evaluation`.

FEVER and RAGTruth results are produced and reported separately. Ground truth is only
read after every prediction for the sample has been made.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.datasets.fever_loader import FeverExample
from src.datasets.ragtruth_loader import RagTruthExample, gold_spans, sentence_labels
from src.evaluation import EvaluationReport, compute_metrics
from src.pipeline import HallucinationPipeline
from src.retrieval.context import ContextRetriever
from src.schemas import LABELS, Answer, Label, RetrievalStatus

Progress = Callable[[int, int], None] | None


@dataclass
class DatasetRun:
    report: EvaluationReport | None
    rows: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    extra_reports: dict[str, EvaluationReport] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def sample(items: list, n: int, seed: int) -> list:
    """Deterministic sample without replacement."""
    if n >= len(items):
        return list(items)
    return random.Random(seed).sample(items, n)


def evaluate_fever(
    examples: list[FeverExample],
    pipeline: HallucinationPipeline,
    n: int,
    seed: int,
    split: str,
    progress: Progress = None,
) -> DatasetRun:
    """Verify FEVER claims with live retrieval; score 3-class predictions.

    Claims whose retrieval *failed* (network error) are excluded from metrics and counted
    separately, because that is an infrastructure failure, not a verifier decision.
    """
    labelled = [e for e in examples if e.label is not None]
    chosen = sample(labelled, n, seed)
    predictions = []
    for i, ex in enumerate(chosen):
        if progress:
            progress(i, len(chosen))
        predictions.append(pipeline.verify_claim_text(ex.claim))  # claim text only

    rows, y_true, y_pred = [], [], []
    n_failed = 0
    for ex, (retrieval, verdict) in zip(chosen, predictions, strict=True):
        failed = retrieval.status is RetrievalStatus.ERROR
        n_failed += failed
        rows.append({
            "id": ex.id, "claim": ex.claim, "gold": ex.label.value, "predicted": verdict.label.value,
            "retrieval": retrieval.status.value,
            "evidence": "; ".join(p.title for p in verdict.evidence),
        })
        if not failed:
            y_true.append(ex.label.value)
            y_pred.append(verdict.label.value)

    warnings = [
        "Evidence comes from today's Wikipedia via live search, not the 2017 dump FEVER was "
        "annotated on, so some gold labels may not match current page content.",
    ]
    if len(labelled) < len(examples):
        warnings.append(f"{len(examples) - len(labelled)} unlabelled rows were ignored.")
    if n_failed:
        warnings.append(f"{n_failed} claims had retrieval failures and were excluded from metrics.")
    config = {
        "dataset": "FEVER", "experiment": "end-to-end (live Wikipedia retrieval)",
        "model": pipeline.verifier.method, "split": split, "sample_size": len(chosen), "seed": seed,
        "verifier": pipeline.verifier.method, "retrieval": "live Wikipedia search",
        "excluded_retrieval_failures": n_failed,
    }
    report = compute_metrics(y_true, y_pred, LABELS, "FEVER", config) if y_true else None
    if report:
        report.warnings = warnings + report.warnings
    else:
        warnings.append("No examples could be scored.")
    return DatasetRun(report, rows, warnings)


HALLUCINATED, NOT_HALLUCINATED = "HALLUCINATED", "NOT_HALLUCINATED"
BINARY = [HALLUCINATED, NOT_HALLUCINATED]


def evaluate_ragtruth(
    examples: list[RagTruthExample],
    pipeline: HallucinationPipeline,
    n: int,
    seed: int,
    split_filter: str | None = "test",
    tasks: tuple[str, ...] | None = ("QA", "Summary"),
    include_implicit_true: bool = False,
    progress: Progress = None,
) -> DatasetRun:
    """Hallucination detection against RAGTruth annotations, at sentence and response level.

    Conversion (see README):
    * evidence is the response's own source context (not Wikipedia);
    * the response is split into sentences with the app's spaCy Sentencizer; a sentence is
      gold HALLUCINATED if an annotated span overlaps it (``implicit_true`` spans excluded
      by default); a response is gold HALLUCINATED if any sentence is;
    * predicted, strict: the sentence's claim(s) were CONTRADICTED by the context;
      predicted, lenient: not SUPPORTED (closer to "baseless info", but also counts misses);
    * sentences the extractor skips (questions, fragments...) and responses without context
      are excluded and counted.
    Data2txt (structured JSON context) is excluded by default because NLI over JSON is poor.
    """
    pool = [e for e in examples if e.context
            and (split_filter is None or e.split == split_filter)
            and (tasks is None or e.task_type in tasks)]
    chosen = sample(pool, n, seed)
    results = []
    for i, ex in enumerate(chosen):
        if progress:
            progress(i, len(chosen))
        answer = Answer(question=ex.prompt[:500], text=ex.response, model=ex.model or "unknown",
                        source="ragtruth")
        results.append(pipeline.analyze(answer, retriever=ContextRetriever(ex.context)))

    sent_gold, sent_strict, sent_lenient = [], [], []
    resp_gold, resp_strict, resp_lenient = [], [], []
    rows: list[dict[str, Any]] = []
    skipped_sentences = 0
    for ex, res in zip(chosen, results, strict=True):
        by_sentence: dict[str, list[Label]] = {}
        for c in res.claims:
            by_sentence.setdefault(c.claim.id.split(".")[0], []).append(c.verdict.label)
        g_any = s_any = l_any = False
        for sent, gold in sentence_labels(ex, include_implicit_true):
            labels = by_sentence.get(sent.id)
            if not labels:
                skipped_sentences += 1
                continue
            strict = Label.CONTRADICTED in labels
            lenient = any(lb is not Label.SUPPORTED for lb in labels)
            sent_gold.append(HALLUCINATED if gold else NOT_HALLUCINATED)
            sent_strict.append(HALLUCINATED if strict else NOT_HALLUCINATED)
            sent_lenient.append(HALLUCINATED if lenient else NOT_HALLUCINATED)
            g_any, s_any, l_any = g_any or gold, s_any or strict, l_any or lenient
        g_resp = bool(gold_spans(ex, include_implicit_true))
        resp_gold.append(HALLUCINATED if g_resp else NOT_HALLUCINATED)
        resp_strict.append(HALLUCINATED if s_any else NOT_HALLUCINATED)
        resp_lenient.append(HALLUCINATED if l_any else NOT_HALLUCINATED)
        labels_all = [c.verdict.label for c in res.claims]
        rows.append({
            "id": ex.id, "task": ex.task_type, "model": ex.model, "gold": resp_gold[-1],
            "n_gold_spans": len(gold_spans(ex, include_implicit_true)), "n_claims": len(res.claims),
            "contradicted": labels_all.count(Label.CONTRADICTED),
            "insufficient": labels_all.count(Label.NOT_ENOUGH_INFORMATION),
            "strict": resp_strict[-1], "lenient": resp_lenient[-1],
        })

    warnings = [
        "RAGTruth annotates character spans for faithfulness to a given context; this app predicts "
        "sentence/claim labels. See the README for the conversion and its limitations.",
    ]
    if skipped_sentences:
        warnings.append(f"{skipped_sentences} sentences were skipped by the extractor and excluded.")
    if not pool or not sent_gold:
        warnings.append("No RAGTruth examples could be scored with the current filters.")
        return DatasetRun(None, rows, warnings)
    config = {
        "dataset": "RAGTruth", "model": pipeline.verifier.method, "split": split_filter or "all",
        "tasks": list(tasks) if tasks else "all", "sample_size": len(chosen), "seed": seed,
        "evidence": "provided source context", "include_implicit_true": include_implicit_true,
        "excluded_skipped_sentences": skipped_sentences,
    }
    reports = {
        "sentence_strict": compute_metrics(sent_gold, sent_strict, BINARY, "RAGTruth sentence-level (strict)",
                                           {**config, "experiment": "C: sentence-level, strict"}),
        "sentence_lenient": compute_metrics(sent_gold, sent_lenient, BINARY, "RAGTruth sentence-level (lenient)",
                                            {**config, "experiment": "C: sentence-level, lenient"}),
        "response_strict": compute_metrics(resp_gold, resp_strict, BINARY, "RAGTruth response-level (strict)",
                                           {**config, "experiment": "C: response-level, strict"}),
        "response_lenient": compute_metrics(resp_gold, resp_lenient, BINARY, "RAGTruth response-level (lenient)",
                                            {**config, "experiment": "C: response-level, lenient"}),
    }
    main = reports.pop("sentence_strict")
    main.warnings = warnings + main.warnings
    return DatasetRun(main, rows, warnings, reports)
