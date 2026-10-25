"""Loader for locally supplied RAGTruth files (https://github.com/ParticleMedia/RAGTruth).

The public release has ``response.jsonl`` (model responses with span annotations) and
``source_info.jsonl`` (the retrieved context and prompt per ``source_id``). Field names
are detected from the data rather than assumed; :func:`inspect_schema` reports what was
found. Annotations are preserved exactly (character spans, label types, extra fields).

Annotation granularity differs from this app: RAGTruth marks character spans as
"Evident/Subtle Conflict" (contradicts the context) or "Evident/Subtle Baseless Info"
(not supported by the context). A span is not the same thing as a contradicted claim, so
evaluation is done at response level, with claim/span overlap reported as a diagnostic.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESPONSE_TEXT_KEYS = ("response", "output", "answer", "generation")
LABEL_LIST_KEYS = ("labels", "hallucination_labels", "annotations", "hallucinations")
CONTEXT_KEYS = ("source_info", "context", "source", "passages", "reference")


@dataclass(frozen=True)
class HallucinationSpan:
    start: int | None
    end: int | None
    text: str
    label_type: str
    extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @property
    def is_conflict(self) -> bool:
        return "conflict" in self.label_type.lower()


@dataclass
class RagTruthExample:
    id: Any
    source_id: Any
    response: str
    context: str
    prompt: str
    task_type: str | None
    model: str | None
    split: str | None
    spans: list[HallucinationSpan]

    @property
    def is_hallucinated(self) -> bool:
        return bool(self.spans)


@dataclass
class SchemaReport:
    response_file: str
    source_file: str | None
    response_keys: list[str]
    source_keys: list[str]
    response_text_key: str | None
    labels_key: str | None
    context_key: str | None
    span_keys: list[str]
    label_types: dict[str, int]
    n_loaded: int = 0
    n_malformed: int = 0
    n_missing_context: int = 0
    splits: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows, errors = [], []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("not a JSON object")
                rows.append(row)
            except ValueError as exc:
                errors.append(f"{path.name} line {i}: {exc}")
    return rows, errors


def _first_key(keys: set[str], candidates: tuple[str, ...]) -> str | None:
    return next((k for k in candidates if k in keys), None)


def context_to_text(value: Any) -> str:
    """Flatten RAGTruth ``source_info`` (string for summaries, dict for QA/data-to-text)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "passages" in value:
            q = value.get("question", "")
            return (f"Question: {q}\n\n" if q else "") + context_to_text(value["passages"])
        return json.dumps(value, ensure_ascii=False, indent=1)
    if isinstance(value, list):
        return "\n\n".join(context_to_text(v) for v in value)
    return str(value)


def inspect_schema(response_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> dict:
    """Detect which field names this copy of RAGTruth uses."""
    r_keys = set().union(*(r.keys() for r in response_rows[:200])) if response_rows else set()
    s_keys = set().union(*(r.keys() for r in source_rows[:200])) if source_rows else set()
    labels_key = _first_key(r_keys, LABEL_LIST_KEYS)
    span_keys: set[str] = set()
    label_types: Counter = Counter()
    for row in response_rows:
        for span in (row.get(labels_key) or []) if labels_key else []:
            if isinstance(span, dict):
                span_keys |= set(span)
                label_types[str(span.get("label_type", span.get("type", "unknown")))] += 1
    return {
        "response_keys": sorted(r_keys),
        "source_keys": sorted(s_keys),
        "response_text_key": _first_key(r_keys, RESPONSE_TEXT_KEYS),
        "labels_key": labels_key,
        "context_key": _first_key(s_keys, CONTEXT_KEYS) or _first_key(r_keys, CONTEXT_KEYS),
        "span_keys": sorted(span_keys),
        "label_types": dict(label_types),
    }


def load_ragtruth(
    response_path: Path, source_path: Path | None = None, limit: int | None = None
) -> tuple[list[RagTruthExample], SchemaReport]:
    """Load responses (and their contexts) with annotations preserved."""
    response_path = Path(response_path)
    if source_path is None:
        guess = response_path.parent / "source_info.jsonl"
        source_path = guess if guess.is_file() else None
    responses, errors = _read_jsonl(response_path)
    sources, s_errors = _read_jsonl(Path(source_path)) if source_path else ([], [])
    schema = inspect_schema(responses, sources)
    report = SchemaReport(
        str(response_path), str(source_path) if source_path else None,
        n_malformed=len(errors) + len(s_errors), errors=(errors + s_errors)[:10], **schema
    )
    text_key, labels_key, ctx_key = schema["response_text_key"], schema["labels_key"], schema["context_key"]
    if text_key is None:
        report.errors.append(f"No response text field found (looked for {RESPONSE_TEXT_KEYS}).")
        return [], report
    source_by_id = {s.get("source_id"): s for s in sources if "source_id" in s}

    examples: list[RagTruthExample] = []
    splits: Counter = Counter()
    for row in responses:
        if limit is not None and len(examples) >= limit:
            break
        text = row.get(text_key)
        if not isinstance(text, str):
            report.n_malformed += 1
            continue
        src = source_by_id.get(row.get("source_id"), {})
        context = context_to_text(src.get(ctx_key) if ctx_key in src else row.get(ctx_key or ""))
        if not context:
            report.n_missing_context += 1
        spans = []
        for span in (row.get(labels_key) or []) if labels_key else []:
            if not isinstance(span, dict):
                continue
            start, end = span.get("start"), span.get("end")
            spans.append(
                HallucinationSpan(
                    start if isinstance(start, int) else None,
                    end if isinstance(end, int) else None,
                    str(span.get("text", "")),
                    str(span.get("label_type", span.get("type", "unknown"))),
                    {k: v for k, v in span.items() if k not in {"start", "end", "text", "label_type"}},
                )
            )
        split = row.get("split")
        splits[str(split)] += 1
        examples.append(
            RagTruthExample(
                row.get("id"), row.get("source_id"), text, context,
                str(src.get("prompt", row.get("prompt", ""))),
                src.get("task_type", row.get("task_type")), row.get("model"),
                split if isinstance(split, str) else None, spans,
            )
        )
    report.n_loaded = len(examples)
    report.splits = dict(splits)
    return examples, report


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


# ------------------------------------------------------------ sentence labels
def gold_spans(example: RagTruthExample, include_implicit_true: bool = False) -> list[HallucinationSpan]:
    """Annotated spans counted as hallucinations.

    Spans marked ``implicit_true`` (correct, just not in the context) are excluded by
    default, because this project checks factual correctness rather than faithfulness.
    """
    return [
        s for s in example.spans
        if include_implicit_true or not s.extra.get("implicit_true", False)
    ]


def sentence_labels(
    example: RagTruthExample, include_implicit_true: bool = False
) -> list[tuple[Any, bool]]:
    """Convert span annotations to sentence labels.

    The response is split with the same spaCy Sentencizer as the app. A sentence is
    labelled hallucinated if any counted span overlaps its character range. Spans without
    offsets are ignored. Returns ``(Sentence, is_hallucinated)`` in order.
    """
    from src.claims.extractor import segment_answer

    spans = [s for s in gold_spans(example, include_implicit_true) if s.start is not None and s.end is not None]
    return [
        (sent, any(spans_overlap(sent.start, sent.end, s.start, s.end) for s in spans))
        for sent in segment_answer(example.response)
    ]


def dataset_statistics(examples: list[RagTruthExample]) -> dict[str, Any]:
    """Counts used for the dataset report: splits, tasks, models, labels, sentence labels."""
    stats: dict[str, Any] = {
        "responses": len(examples),
        "by_split": dict(Counter(str(e.split) for e in examples)),
        "by_task": dict(Counter(str(e.task_type) for e in examples)),
        "by_model": dict(Counter(str(e.model) for e in examples)),
        "responses_with_any_span": sum(bool(e.spans) for e in examples),
        "responses_with_counted_span": sum(bool(gold_spans(e)) for e in examples),
        "span_label_types": dict(Counter(s.label_type for e in examples for s in e.spans)),
        "spans_implicit_true": sum(bool(s.extra.get("implicit_true")) for e in examples for s in e.spans),
        "missing_context": sum(not e.context for e in examples),
    }
    return stats
