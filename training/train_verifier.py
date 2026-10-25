"""Fine-tune a pretrained sequence classifier on FEVER claim-evidence pairs.

    python -m training.train_verifier --base-model microsoft/deberta-v3-base \
        --data data/fever/prepared --out models/fever-verifier --epochs 1 --batch-size 8

Input format (same at training, evaluation and inference): a text pair
``(evidence, claim)``, i.e. evidence as the premise and the claim as the hypothesis.
The FEVER label is only the target; it is never part of the input.

If the base model is already an NLI classifier, its entailment/contradiction/neutral
head is reused (SUPPORTS -> entailment, REFUTES -> contradiction, NEI -> neutral);
otherwise a new 3-way head is initialised.

The best checkpoint on the validation split (held out from official train) is saved with
its tokenizer and ``training_record.json`` (base model, arguments, seed, data counts,
per-epoch validation metrics, duration, hardware). A failed run is recorded as failed and
is never saved as a usable checkpoint.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
NLI_NAME = {"SUPPORTS": "entailment", "REFUTES": "contradiction", "NOT ENOUGH INFO": "neutral"}


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_mapping(config) -> tuple[dict[str, int], bool]:
    """FEVER label -> class index; reuse an existing NLI head when there is one."""
    from src.verification.nli_verifier import normalise_label_name

    names = {int(i): normalise_label_name(str(n)) for i, n in (config.id2label or {}).items()}
    if getattr(config, "num_labels", 0) == 3 and set(names.values()) == {"entailment", "contradiction", "neutral"}:
        inv = {v: k for k, v in names.items()}
        return {lab: inv[NLI_NAME[lab]] for lab in LABELS}, True
    return {lab: i for i, lab in enumerate(LABELS)}, False


def evaluate(model, tokenizer, pairs, label2id, batch_size, max_length, device) -> dict:
    import torch

    from src.evaluation import compute_metrics

    id2fever = {v: k for k, v in label2id.items()}
    preds: list[str] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            enc = tokenizer([p.evidence for p in batch], [p.claim for p in batch], truncation="only_first",
                            max_length=max_length, padding=True, return_tensors="pt").to(device)
            preds += [id2fever[int(j)] for j in model(**enc).logits.argmax(-1).tolist()]
    report = compute_metrics([p.label for p in pairs], preds, LABELS, "FEVER validation (training)")
    return {"n": report.n, "accuracy": report.accuracy, "macro_f1": report.macro_f1,
            "per_class": report.per_class, "confusion_matrix": report.confusion_matrix}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="microsoft/deberta-v3-base")
    ap.add_argument("--data", type=Path, default=Path("data/fever/prepared"))
    ap.add_argument("--out", type=Path, default=Path("models/fever-verifier"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-train", type=int, default=0, help="cap on training pairs (0 = all prepared)")
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--allow-download", action="store_true",
                    help="allow downloading the base model from Hugging Face (deberta-v3-base is ~0.7 GB)")
    ap.add_argument("--cache-dir", type=Path, default=Path("models/hf-cache"))
    ap.add_argument("--log-every", type=int, default=50, help="batches between progress lines")
    args = ap.parse_args(argv)

    from src.datasets.fever_pairs import read_pairs

    record: dict = {"status": "started", "base_model": args.base_model,
                    "args": {k: str(v) for k, v in vars(args).items()},
                    "input_format": "text pair (evidence, claim); label is the target only",
                    "seed": args.seed}
    run_dir = args.out
    run_dir.mkdir(parents=True, exist_ok=True)
    for old in run_dir.glob("training_record*.json"):  # never leave a stale record behind
        old.unlink()
    progress_path = run_dir / "progress.log"
    progress_path.write_text("", encoding="utf-8")

    def log(msg: str) -> None:
        """Progress goes to stderr and to progress.log, so a run can be checked from outside."""
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, file=sys.stderr, flush=True)
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    t0 = time.time()
    try:
        import torch
        import transformers
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )

        set_seed(args.seed)
        train = read_pairs(args.data / "train.jsonl")
        val = read_pairs(args.data / "val.jsonl")
        rng = random.Random(args.seed)
        rng.shuffle(train)
        if args.max_train:
            train = train[:args.max_train]
        if args.max_val:
            val = val[:args.max_val]
        overlap = {p.claim for p in train} & {p.claim for p in val}
        if overlap:
            raise RuntimeError(f"{len(overlap)} claims appear in both train and val; re-run prepare_fever.")
        record["data"] = {"train_pairs": len(train), "val_pairs": len(val),
                          "train_classes": dict(Counter(p.label for p in train)),
                          "val_classes": dict(Counter(p.label for p in val)),
                          "prepare_report": str(args.data / "prepare_report.json")}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        record["hardware"] = {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                              "cpu": platform.processor() or platform.machine(), "torch": torch.__version__,
                              "transformers": transformers.__version__, "python": platform.python_version()}
        kw = {"local_files_only": not args.allow_download}
        if not Path(args.base_model).is_dir():
            kw["cache_dir"] = str(args.cache_dir)
        config = AutoConfig.from_pretrained(args.base_model, **kw)
        label2id, reused = label_mapping(config)
        record["head"] = "reused NLI head" if reused else "new 3-way head"
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, **kw)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model, num_labels=3, id2label={v: k for k, v in label2id.items()}, label2id=label2id,
            ignore_mismatched_sizes=not reused, dtype=torch.float32, **kw)  # fp32 master weights; AMP casts
        model.to(device)

        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        per_step = args.batch_size * args.grad_accum
        steps_per_epoch = max(1, (len(train) + per_step - 1) // per_step)
        total = steps_per_epoch * args.epochs
        sched = get_linear_schedule_with_warmup(optim, int(total * args.warmup_ratio), total)
        use_amp = device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        batches_per_epoch = (len(train) + args.batch_size - 1) // args.batch_size
        log(f"device {device}; {len(train)} train / {len(val)} val pairs; "
            f"{batches_per_epoch} batches per epoch x {args.epochs} epoch(s)")
        best_f1, epochs = -1.0, []
        for epoch in range(1, args.epochs + 1):
            model.train()
            running, n_batches = 0.0, 0
            t_epoch = time.time()
            for i in range(0, len(train), args.batch_size):
                batch = train[i:i + args.batch_size]
                enc = tokenizer([p.evidence for p in batch], [p.claim for p in batch], truncation="only_first",
                                max_length=args.max_length, padding=True, return_tensors="pt").to(device)
                labels = torch.tensor([label2id[p.label] for p in batch], device=device)
                with torch.autocast(device_type=device, enabled=use_amp):
                    loss = model(**enc, labels=labels).loss / args.grad_accum
                scaler.scale(loss).backward()
                n_batches += 1
                running += loss.item() * args.grad_accum
                if n_batches % args.grad_accum == 0 or i + args.batch_size >= len(train):
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()
                    sched.step()
                if n_batches % args.log_every == 0:
                    per_batch = (time.time() - t_epoch) / n_batches
                    eta = per_batch * (batches_per_epoch - n_batches)
                    log(f"epoch {epoch} batch {n_batches}/{batches_per_epoch} loss {running / n_batches:.4f} "
                        f"{per_batch:.2f}s/batch, epoch ETA {eta / 60:.1f} min")
            log(f"epoch {epoch} training done in {(time.time() - t_epoch) / 60:.1f} min; validating {len(val)} pairs")
            val_metrics = evaluate(model, tokenizer, val, label2id, args.batch_size * 2, args.max_length, device)
            epochs.append({"epoch": epoch, "train_loss": running / max(n_batches, 1), "val": val_metrics})
            log(f"epoch {epoch}: val macro F1 {val_metrics['macro_f1']:.4f}")
            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                model.save_pretrained(run_dir)
                tokenizer.save_pretrained(run_dir)
                record["best_epoch"] = epoch
        record.update({"status": "completed", "epochs": epochs, "best_val_macro_f1": best_f1})
    except Exception as exc:  # noqa: BLE001 - record the failure honestly
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()[-2000:]})
    record["duration_seconds"] = round(time.time() - t0, 1)
    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    name = "training_record.json" if record["status"] == "completed" else "training_record.failed.json"
    (run_dir / name).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("status", "duration_seconds") if k in record}), file=sys.stderr)
    if record["status"] != "completed":
        print(record.get("error"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
