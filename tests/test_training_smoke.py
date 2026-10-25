"""End-to-end smoke test of training, evaluation and NLI loading with real torch/transformers.

Uses a tiny randomly initialised BERT built locally (no download). It checks that the code
runs and writes its records; the resulting "model" is meaningless and nothing about its
metrics is a result. Skipped when torch/transformers are not installed.
"""

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

PAIRS = [
    ("SUPPORTS", "Paris is in France.", "Paris: Paris is the capital of France."),
    ("REFUTES", "Paris is in Spain.", "Paris: Paris is the capital of France."),
    ("NOT ENOUGH INFO", "Paris has a famous bakery.", "Paris: Paris is the capital of France."),
    ("SUPPORTS", "Rome is in Italy.", "Rome: Rome is the capital of Italy."),
    ("REFUTES", "Rome is in Peru.", "Rome: Rome is the capital of Italy."),
    ("NOT ENOUGH INFO", "Rome has many cats.", "Rome: Rome is the capital of Italy."),
]


def tiny_model(path):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    text = " ".join(c + " " + e for _, c, e in PAIRS).lower().replace(".", " .").replace(":", " :")
    words = sorted(set(text.split()))
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]
    path.mkdir()
    (path / "vocab.txt").write_text("\n".join(vocab))
    tok = BertTokenizerFast(vocab_file=str(path / "vocab.txt"))
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=64)
    BertForSequenceClassification(cfg).save_pretrained(path)
    tok.save_pretrained(path)


def write_pairs(path, name):
    with (path / f"{name}.jsonl").open("w") as fh:
        for i, (label, claim, ev) in enumerate(PAIRS):
            claim = claim if name == "train" else claim.replace(".", " indeed.")
            fh.write(json.dumps({"id": i, "claim": claim, "evidence": ev, "label": label, "split": name,
                                 "evidence_source": "gold", "pages": []}) + "\n")


def test_train_evaluate_and_load_tiny_model(tmp_path):
    from src.verification.nli_verifier import NLIVerifier
    from training.evaluate_verifier import main as evaluate_main
    from training.train_verifier import main as train_main

    base = tmp_path / "tiny-base"
    tiny_model(base)
    data = tmp_path / "prepared"
    data.mkdir()
    write_pairs(data, "train")
    write_pairs(data, "val")
    out = tmp_path / "fever-verifier"
    assert train_main(["--base-model", str(base), "--data", str(data), "--out", str(out), "--epochs", "2",
                       "--batch-size", "2", "--grad-accum", "1", "--max-length", "32", "--lr", "1e-3"]) == 0
    record = json.loads((out / "training_record.json").read_text())
    assert record["status"] == "completed" and record["head"] == "new 3-way head"
    assert record["data"]["train_pairs"] == 6 and len(record["epochs"]) == 2
    assert record["hardware"]["device"] in {"cpu", "cuda"} and record["duration_seconds"] >= 0

    verifier = NLIVerifier(str(out))
    assert "fine-tuned on FEVER by this project" in verifier.method
    verifier.load()
    from src.schemas import EvidencePassage, Label
    v = verifier.verify("Paris is in France.", [EvidencePassage("Paris", "u", "Paris is the capital of France.")])
    assert v.label in set(Label) and len(v.raw_scores["per_passage"]) == 1
    scores = v.raw_scores["per_passage"][0]
    assert abs(scores["entailment"] + scores["neutral"] + scores["contradiction"] - 1) < 1e-3

    results = tmp_path / "results"
    assert evaluate_main(["--model", str(out), "--data", str(data), "--split", "val", "--out", str(results)]) == 0
    report = json.loads(next(results.glob("fever_*.json")).read_text())
    assert report["config"]["experiment"] == "B: fine-tuned" and report["n"] == 6


def test_failed_training_is_recorded_not_saved(tmp_path):
    from training.train_verifier import main as train_main

    data = tmp_path / "prepared"
    data.mkdir()
    write_pairs(data, "train")
    write_pairs(data, "val")
    out = tmp_path / "run"
    assert train_main(["--base-model", str(tmp_path / "does-not-exist"), "--data", str(data), "--out", str(out)]) == 1
    assert (out / "training_record.failed.json").is_file()
    assert not (out / "training_record.json").exists()
