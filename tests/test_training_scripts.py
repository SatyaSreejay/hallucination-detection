"""Training/evaluation script logic that does not need model weights."""

import json
from types import SimpleNamespace

from training.evaluate_verifier import main as evaluate_main
from training.evaluate_verifier import tune
from training.train_verifier import label_mapping


def test_label_mapping_reuses_nli_head_in_its_own_order():
    nli = SimpleNamespace(num_labels=3, id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
    mapping, reused = label_mapping(nli)
    assert reused and mapping == {"SUPPORTS": 1, "REFUTES": 0, "NOT ENOUGH INFO": 2}
    plain = SimpleNamespace(num_labels=2, id2label={0: "LABEL_0", 1: "LABEL_1"})
    mapping, reused = label_mapping(plain)
    assert not reused and mapping == {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}


def test_threshold_tuning_finds_known_optimum():
    gold = ["SUPPORTED", "NOT_ENOUGH_INFORMATION", "CONTRADICTED"]
    scores = [{"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05},
              {"entailment": 0.55, "neutral": 0.4, "contradiction": 0.05},  # weak entailment, gold NEI
              {"entailment": 0.1, "neutral": 0.1, "contradiction": 0.8}]
    sup, con, f1 = tune(gold, scores)
    assert f1 == 1.0 and sup is not None and 0.55 < sup <= 0.9


def _pairs(path):
    rows = [{"id": 1, "claim": "A is B.", "evidence": "A: A is B.", "label": "SUPPORTS", "split": "x",
             "evidence_source": "gold", "pages": ["A"]},
            {"id": 2, "claim": "A is C.", "evidence": "A: A is B.", "label": "REFUTES", "split": "x",
             "evidence_source": "gold", "pages": ["A"]}]
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_evaluate_script_refuses_tuning_on_held_out_split(tmp_path):
    (tmp_path / "dev.jsonl").write_text("")
    assert evaluate_main(["--model", "m", "--data", str(tmp_path), "--split", "dev", "--tune"]) == 2


def test_evaluate_script_with_stubbed_model(tmp_path, monkeypatch):
    from src.verification import nli_verifier

    _pairs(tmp_path / "val.jsonl")

    def fake_scores(self, premises, hypotheses, batch_size=16):
        return [{"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05} if h == "A is B." else
                {"entailment": 0.05, "neutral": 0.05, "contradiction": 0.9} for h in hypotheses]
    monkeypatch.setattr(nli_verifier.NLIVerifier, "score_pairs", fake_scores)
    out = tmp_path / "results"
    assert evaluate_main(["--model", "stub/model", "--data", str(tmp_path), "--split", "val", "--tune",
                          "--out", str(out)]) == 0
    report = json.loads(next(out.glob("fever_*.json")).read_text())
    assert report["n"] == 2 and report["accuracy"] == 1.0
    assert report["config"]["experiment"] == "A: pretrained baseline"
    assert any("Validation split" in w for w in report["warnings"])
    assert (out / "thresholds_model.json").is_file()
