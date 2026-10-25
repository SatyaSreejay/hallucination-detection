"""Prepare FEVER claim-evidence pairs for training and evaluating the verifier.

    python -m training.prepare_fever --fever-dir data/fever --wiki-dir data/fever/wiki-pages \
        --max-per-class 3000 --val-fraction 0.1

Inputs (supplied by you, see data/README.md):
  data/fever/train.jsonl                         official training split (required)
  data/fever/shared_task_dev.jsonl or paper_dev.jsonl / paper_test.jsonl  labelled held-out splits
  data/fever/wiki-pages/wiki-*.jsonl             FEVER Wikipedia dump (needed for evidence text)

Outputs in data/fever/prepared/: train.jsonl, val.jsonl (held out from official train),
<dev split>.jsonl, and prepare_report.json with real per-split/per-class counts.
Without the wiki dump only the statistics report is written.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from src.datasets.fever_loader import load_fever_jsonl
from src.datasets.fever_pairs import (
    balanced_subset,
    build_pairs,
    claim_key,
    class_counts,
    load_pages,
    needed_pages,
    split_train_validation,
)

HELD_OUT_FILES = ["shared_task_dev.jsonl", "paper_dev.jsonl", "paper_test.jsonl"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fever-dir", type=Path, default=Path("data/fever"))
    ap.add_argument("--wiki-dir", type=Path, default=Path("data/fever/wiki-pages"))
    ap.add_argument("--out", type=Path, default=Path("data/fever/prepared"))
    ap.add_argument("--max-per-class", type=int, default=3000,
                    help="cap per class for the training subset (use 0 for no cap)")
    ap.add_argument("--max-per-class-eval", type=int, default=1000,
                    help="cap per class for validation and held-out splits (0 = no cap)")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)
    cap = args.max_per_class or None
    cap_eval = args.max_per_class_eval or None
    t0 = time.time()

    train_path = args.fever_dir / "train.jsonl"
    if not train_path.is_file():
        print(f"Missing {train_path}. Download train.jsonl from https://fever.ai/dataset/fever.html", file=sys.stderr)
        return 2
    report: dict = {"seed": args.seed, "max_per_class_train": cap, "max_per_class_eval": cap_eval,
                    "val_fraction": args.val_fraction, "files": {}, "splits": {}}

    train_all, rep = load_fever_jsonl(train_path)
    report["files"]["train"] = {"path": str(train_path), "rows": rep.n_loaded, "malformed": rep.n_malformed,
                                "unlabelled": rep.n_unlabelled, "classes": class_counts(train_all)}
    train_ex, val_ex = split_train_validation(train_all, args.val_fraction, args.seed)
    splits = {"train": balanced_subset(train_ex, cap, args.seed),
              "val": balanced_subset(val_ex, cap_eval, args.seed)}
    pools = {"train": train_ex, "val": val_ex}

    train_keys = {claim_key(e.claim) for e in train_ex}
    for name in HELD_OUT_FILES:
        path = args.fever_dir / name
        if not path.is_file():
            continue
        exs, rep = load_fever_jsonl(path)
        overlap = sum(claim_key(e.claim) in train_keys for e in exs)
        report["files"][path.stem] = {"path": str(path), "rows": rep.n_loaded, "malformed": rep.n_malformed,
                                      "unlabelled": rep.n_unlabelled, "classes": class_counts(exs),
                                      "claim_text_overlap_with_train": overlap}
        exs = [e for e in exs if claim_key(e.claim) not in train_keys]  # no leakage
        splits[path.stem] = balanced_subset(exs, cap_eval, args.seed)
        pools[path.stem] = exs
    if len(splits) == 2:
        report["warning"] = ("No official labelled held-out split found (shared_task_dev / paper_dev / "
                             "paper_test). Final evaluation needs one of them.")

    for name, exs in splits.items():
        report["splits"][name] = {"available_examples": len(pools[name]),
                                  "available_classes": class_counts(pools[name]),
                                  "selected_examples": len(exs), "classes": class_counts(exs)}

    if not args.wiki_dir.is_dir() or not any(args.wiki_dir.glob("*.jsonl")):
        report["pairs_built"] = False
        report["note"] = (f"FEVER Wikipedia dump not found in {args.wiki_dir}; evidence text cannot be built. "
                          "Download wiki-pages.zip from https://fever.ai/dataset/fever.html and unzip it there.")
        _write_report(args.out, report, t0)
        print(json.dumps(report, indent=2))
        return 1

    wanted = set().union(*(needed_pages(exs) for exs in splits.values()))
    print(f"Scanning wiki dump for {len(wanted)} candidate pages...", file=sys.stderr)
    pages = load_pages(args.wiki_dir, wanted)
    args.out.mkdir(parents=True, exist_ok=True)
    report["pairs_built"] = True
    report["pages_found"] = len(pages)
    for name, exs in splits.items():
        pairs, dropped = build_pairs(exs, pages, name, args.seed)
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for p in pairs:
                fh.write(p.to_json() + "\n")
        report["splits"][name].update({
            "pairs": len(pairs),
            "pair_classes": class_counts(pairs),
            "dropped": dict(dropped),
            "evidence_sources": dict(Counter(p.evidence_source for p in pairs)),
        })
    _write_report(args.out, report, t0)
    print(json.dumps(report, indent=2))
    return 0


def _write_report(out: Path, report: dict, t0: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    report["seconds"] = round(time.time() - t0, 1)
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
