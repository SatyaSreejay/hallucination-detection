"""Metrics over real ground-truth labels (scikit-learn).

:func:`compute_metrics` refuses to run without a ground-truth label for every example,
so metrics can only ever come from labelled data (FEVER, RAGTruth annotations, or the
expected labels of validated controlled mutations).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

SMALL_SAMPLE_WARNING_N = 30


@dataclass
class EvaluationReport:
    name: str
    n: int
    labels: list[str]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]  # rows = true label, columns = predicted label
    true_distribution: dict[str, int]
    pred_distribution: dict[str, int]
    config: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, directory: Path) -> Path:
        """Write the report as JSON and return its path."""
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.created_at.replace(":", "").replace("-", "")
        slug = "".join(c if c.isalnum() else "_" for c in self.name.lower())
        path = directory / f"{slug}_{stamp}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    name: str,
    config: dict[str, Any] | None = None,
) -> EvaluationReport:
    """Accuracy, per-class and macro precision/recall/F1, and a confusion matrix.

    Raises ``ValueError`` on missing or unknown ground-truth labels.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different lengths")
    if not y_true:
        raise ValueError("No labelled examples to evaluate")
    unknown = {y for y in y_true if y not in labels}
    if unknown:
        raise ValueError(f"Ground truth contains missing/unknown labels: {unknown}")
    bad_pred = {y for y in y_pred if y not in labels}
    if bad_pred:
        raise ValueError(f"Predictions contain unknown labels: {bad_pred}")

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    true_dist = Counter(y_true)
    present = [i for i, lab in enumerate(labels) if true_dist[lab] > 0]
    warnings: list[str] = []
    if len(y_true) < SMALL_SAMPLE_WARNING_N:
        warnings.append(
            f"Only {len(y_true)} examples: metrics are highly uncertain and should not be "
            "generalised."
        )
    missing = [lab for lab in labels if true_dist[lab] == 0]
    if missing:
        warnings.append(
            f"No ground-truth examples for {missing}; recall for those classes is undefined "
            "and macro averages use only the classes present."
        )

    def macro(values) -> float:
        return float(sum(values[i] for i in present) / len(present)) if present else 0.0

    return EvaluationReport(
        name=name,
        n=len(y_true),
        labels=list(labels),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=macro(p),
        macro_recall=macro(r),
        macro_f1=macro(f),
        per_class={
            lab: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i, lab in enumerate(labels)
        },
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        true_distribution={lab: true_dist[lab] for lab in labels},
        pred_distribution={lab: Counter(y_pred)[lab] for lab in labels},
        config=config or {},
        warnings=warnings,
    )
