"""Metric calculations on known labels."""

import pytest

from src.evaluation import compute_metrics
from src.schemas import LABELS


def test_known_three_class_metrics(tmp_path):
    y_true = ["SUPPORTED", "SUPPORTED", "CONTRADICTED", "CONTRADICTED", "NOT_ENOUGH_INFORMATION"]
    y_pred = ["SUPPORTED", "CONTRADICTED", "CONTRADICTED", "CONTRADICTED", "SUPPORTED"]
    r = compute_metrics(y_true, y_pred, LABELS, "unit", {"split": "fixture"})
    assert r.n == 5 and r.accuracy == pytest.approx(0.6)
    sup = r.per_class["SUPPORTED"]
    assert sup["precision"] == pytest.approx(0.5) and sup["recall"] == pytest.approx(0.5)
    con = r.per_class["CONTRADICTED"]
    assert con["precision"] == pytest.approx(2 / 3) and con["recall"] == 1.0
    assert con["f1"] == pytest.approx(0.8)
    assert r.per_class["NOT_ENOUGH_INFORMATION"]["recall"] == 0.0
    assert r.confusion_matrix == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    assert r.true_distribution == {"SUPPORTED": 2, "CONTRADICTED": 2, "NOT_ENOUGH_INFORMATION": 1}
    assert r.macro_f1 == pytest.approx((0.5 + 0.8 + 0.0) / 3)
    assert any("Only 5 examples" in w for w in r.warnings)
    assert r.config["split"] == "fixture"
    path = r.save(tmp_path)
    assert path.exists() and '"accuracy": 0.6' in path.read_text()


def test_absent_class_is_warned_and_excluded_from_macro():
    r = compute_metrics(["SUPPORTED", "CONTRADICTED"], ["SUPPORTED", "CONTRADICTED"], LABELS, "u")
    assert r.macro_f1 == 1.0
    assert any("No ground-truth examples" in w for w in r.warnings)


def test_refuses_invented_or_missing_labels():
    with pytest.raises(ValueError):
        compute_metrics([None], ["SUPPORTED"], LABELS, "u")
    with pytest.raises(ValueError):
        compute_metrics([], [], LABELS, "u")
    with pytest.raises(ValueError):
        compute_metrics(["SUPPORTED"], [], LABELS, "u")
    with pytest.raises(ValueError):
        compute_metrics(["SUPPORTED"], ["MAYBE"], LABELS, "u")
