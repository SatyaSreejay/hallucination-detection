"""Loader for locally supplied FEVER JSONL files (https://fever.ai/dataset/fever.html).

Supports the official nested format::

    {"id": 75397, "verifiable": "VERIFIABLE", "label": "SUPPORTS", "claim": "...",
     "evidence": [[[92206, 104971, "Nikolaj_Coster-Waldau", 7], ...], ...]}

and the flattened format used by some mirrors (``evidence_annotation_id``,
``evidence_id``, ``evidence_wiki_url``, ``evidence_sentence_id``). Labels are mapped to the
application's schema. Rows without a label (e.g. the blind test set) are kept with
``label=None`` and are never used for metrics.

FEVER evidence is a list of Wikipedia (2017 dump) page ids and sentence numbers, not text.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.schemas import Label

logger = logging.getLogger(__name__)

FEVER_LABEL_MAP: dict[str, Label] = {
    "SUPPORTS": Label.SUPPORTED,
    "SUPPORTED": Label.SUPPORTED,
    "REFUTES": Label.CONTRADICTED,
    "REFUTED": Label.CONTRADICTED,
    "NOT ENOUGH INFO": Label.NOT_ENOUGH_INFORMATION,
    "NOT_ENOUGH_INFO": Label.NOT_ENOUGH_INFORMATION,
    "NOT ENOUGH INFORMATION": Label.NOT_ENOUGH_INFORMATION,
    "NEI": Label.NOT_ENOUGH_INFORMATION,
}


def normalize_fever_label(raw: Any) -> Label | None:
    """Map a FEVER label string to :class:`Label`; ``None`` if missing/unknown."""
    if not isinstance(raw, str):
        return None
    return FEVER_LABEL_MAP.get(raw.strip().upper().replace("-", " "))


@dataclass(frozen=True)
class FeverEvidence:
    annotation_id: int | None
    evidence_id: int | None
    wiki_page: str | None
    sentence_id: int | None


@dataclass
class FeverExample:
    id: Any
    claim: str
    label: Label | None
    original_label: str | None
    evidence_sets: list[list[FeverEvidence]]
    verifiable: str | None = None


@dataclass
class LoadReport:
    path: str
    split: str
    n_loaded: int = 0
    n_malformed: int = 0
    n_unlabelled: int = 0
    unknown_labels: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)  # first few problems, for display


def infer_split(path: Path) -> str:
    """Best-effort split name from the file name (train / dev / test)."""
    name = path.stem.lower()
    for split in ("train", "dev", "test"):
        if split in name:
            return name if name != split else split
    return name


def _parse_evidence(row: dict[str, Any]) -> list[list[FeverEvidence]]:
    def as_int(v: Any) -> int | None:
        return v if isinstance(v, int) else None

    evidence = row.get("evidence")
    sets: list[list[FeverEvidence]] = []
    if isinstance(evidence, list):
        for ev_set in evidence:
            if not isinstance(ev_set, list):
                continue
            items = []
            for ev in ev_set:
                if isinstance(ev, list) and len(ev) >= 4:
                    items.append(
                        FeverEvidence(as_int(ev[0]), as_int(ev[1]),
                                      ev[2] if isinstance(ev[2], str) else None, as_int(ev[3]))
                    )
            if items:
                sets.append(items)
    elif "evidence_wiki_url" in row:  # flattened mirror format: one evidence item per row
        sets.append([
            FeverEvidence(
                as_int(row.get("evidence_annotation_id")),
                as_int(row.get("evidence_id")),
                row.get("evidence_wiki_url") or None,
                as_int(row.get("evidence_sentence_id")),
            )
        ])
    return sets


def load_fever_jsonl(path: Path, limit: int | None = None) -> tuple[list[FeverExample], LoadReport]:
    """Read a FEVER JSONL file, skipping (and reporting) malformed rows."""
    path = Path(path)
    report = LoadReport(str(path), infer_split(path))
    examples: list[FeverExample] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if limit is not None and len(examples) >= limit:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row is not a JSON object")
                claim = row.get("claim")
                if not isinstance(claim, str) or not claim.strip():
                    raise ValueError("missing or empty 'claim'")
            except ValueError as exc:
                report.n_malformed += 1
                if len(report.errors) < 10:
                    report.errors.append(f"line {line_no}: {exc}")
                continue
            raw_label = row.get("label")
            label = normalize_fever_label(raw_label)
            if raw_label is None:
                report.n_unlabelled += 1
            elif label is None:
                report.unknown_labels[str(raw_label)] = report.unknown_labels.get(str(raw_label), 0) + 1
            examples.append(
                FeverExample(row.get("id"), claim.strip(), label,
                             raw_label if isinstance(raw_label, str) else None,
                             _parse_evidence(row), row.get("verifiable"))
            )
    report.n_loaded = len(examples)
    return examples, report
