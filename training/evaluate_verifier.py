"""Evaluate a verifier checkpoint on prepared FEVER pairs (gold/oracle evidence setting).

Experiment A (pretrained baseline):
    python -m training.evaluate_verifier --model cross-encoder/nli-deberta-v3-xsmall --split shared_task_dev
Experiment B (fine-tuned):
    python -m training.evaluate_verifier --model models/fever-verifier --split shared_task_dev
Threshold search (validation split only; writes a thresholds file):
    python -m training.evaluate_verifier --model ... --split val --tune
    python -m training.evaluate_verifier --model ... --split shared_task_dev \
        --thresholds results/thresholds_<model>.json

Predictions use exactly the label mapping used by the app (:func:`label_from_scores`):
entailment -> SUPPORTED, contradiction -> CONTRADICTED, neutral -> NOT_ENOUGH_INFORMATION,
with argmax unless validated thresholds are supplied. The metrics therefore also measure how
well that NLI-to-FEVER mapping works. Results are written to results/*.json.

This measures the verifier with FEVER's annotated evidence (NEI evidence from the nearest
page heuristic). It is not an end-to-end score with live retrieval; for that use the
FEVER tab of the dashboard or ``python -m src.cli eval-fever``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

from src.config import RESULTS_DIR
from src.datasets.fever_pairs import read_pairs
from src.evaluation import compute_metrics
from src.schemas import LABELS, Label
from src.verification.nli_verifier import NLIVerifier, label_from_scores

FEVER_TO_APP = {"SUPPORTS": Label.SUPPORTED.value, "REFUTES": Label.CONTRADICTED.value,
                "NOT ENOUGH INFO": Label.NOT_ENOUGH_INFORMATION.value}
GRID = [None, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def predict(scores, sup, con) -> list[str]:
    return [label_from_scores(s, sup, con).value for s in scores]


def tune(gold: list[str], scores: list[dict]) -> tuple[float | None, float | None, float]:
    """Grid-search minimum scores that maximise macro F1 on validation data."""
    best = (None, None, -1.0)
    for sup, con in itertools.product(GRID, GRID):
        f1 = compute_metrics(gold, predict(scores, sup, con), LABELS, "tune").macro_f1
        if f1 > best[2] + 1e-9:
            best = (sup, con, f1)
    return best


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.strip("/").split("/")[-1].lower())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face id (cached) or local checkpoint directory")
    ap.add_argument("--data", type=Path, default=Path("data/fever/prepared"))
    ap.add_argument("--split", default="val", help="val, or an official held-out split such as shared_task_dev")
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tune", action="store_true", help="search thresholds (only on --split val)")
    ap.add_argument("--thresholds", type=Path, help="JSON from a previous --tune run")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--cache-dir", type=Path, default=Path("models/hf-cache"))
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)

    if args.tune and args.split != "val":
        print("Refusing to tune thresholds on a held-out evaluation split; use --split val.", file=sys.stderr)
        return 2
    path = args.data / f"{args.split}.jsonl"
    if not path.is_file():
        print(f"Missing {path}. Run `python -m training.prepare_fever` first.", file=sys.stderr)
        return 2
    pairs = read_pairs(path)
    if args.max_examples:
        pairs = pairs[: args.max_examples]

    sup = con = None
    if args.thresholds:
        th = json.loads(args.thresholds.read_text())
        if th.get("tuned_on_split") != "val":
            print("Thresholds file was not tuned on the validation split; refusing.", file=sys.stderr)
            return 2
        sup, con = th["support_threshold"], th["contradiction_threshold"]

    verifier = NLIVerifier(args.model, cache_dir=args.cache_dir, allow_download=args.allow_download)
    t0 = time.time()
    scores = verifier.score_pairs([p.evidence for p in pairs], [p.claim for p in pairs], args.batch_size)
    seconds = round(time.time() - t0, 1)
    gold = [FEVER_TO_APP[p.label] for p in pairs]
    record = verifier.training_record
    kind = "B: fine-tuned" if record and record.get("status") == "completed" else "A: pretrained baseline"
    args.out.mkdir(parents=True, exist_ok=True)

    if args.tune:
        sup, con, f1 = tune(gold, scores)
        th_path = args.out / f"thresholds_{slug(args.model)}.json"
        th_path.write_text(json.dumps({"model": args.model, "tuned_on_split": "val", "n": len(pairs),
                                       "support_threshold": sup, "contradiction_threshold": con,
                                       "val_macro_f1": f1, "grid": GRID}, indent=2))
        print(f"Best thresholds on val: support={sup} contradiction={con} (macro F1 {f1:.4f}) -> {th_path}",
              file=sys.stderr)

    report = compute_metrics(gold, predict(scores, sup, con), LABELS, f"FEVER ({kind})", {
        "dataset": "FEVER", "experiment": kind, "model": args.model, "split": args.split,
        "evidence_setting": "annotated gold evidence; NEI evidence from nearest-page heuristic",
        "label_mapping": "entailment->SUPPORTED, contradiction->CONTRADICTED, neutral->NOT_ENOUGH_INFORMATION",
        "decision_rule": "argmax" if sup is None and con is None else f"thresholds support={sup} contradiction={con}",
        "thresholds_tuned_here": bool(args.tune), "inference_seconds": seconds,
        "data_file": str(path), "training_record": record,
    })
    if args.split == "val":
        report.warnings.insert(0, "Validation split (held out from official train) is used for model selection "
                               "and threshold tuning; report results on an official held-out split as final.")
    saved = report.save(args.out)
    print(json.dumps({"saved": str(saved), "n": report.n, "accuracy": report.accuracy,
                      "macro_f1": report.macro_f1, "warnings": report.warnings}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
