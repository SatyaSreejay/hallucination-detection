"""Build (claim, evidence text, label) pairs from FEVER + the FEVER Wikipedia dump.

FEVER claims only reference evidence as ``(page_id, sentence_number)`` in the 2017
Wikipedia dump (``wiki-pages.zip`` from fever.ai, about 1.7 GB unzipped into
``wiki-*.jsonl`` files). This module streams those files and keeps only the pages it
needs.

Evidence construction:
* SUPPORTS / REFUTES: sentences of the first annotated evidence set, prefixed by the page
  title ("gold" evidence). Examples whose sentences are missing from the dump are dropped
  and counted.
* NOT ENOUGH INFO has no annotated evidence. Following common practice (NearestPage-style
  sampling), evidence is sentences from a page whose title matches a capitalised n-gram of
  the claim ("nearest_page"). Claims with no matching page are dropped and counted.

The label is only ever the training target; it is never part of the model input.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from src.datasets.fever_loader import FeverExample
from src.schemas import Label

FEVER_LABEL_NAMES = {
    Label.SUPPORTED: "SUPPORTS",
    Label.CONTRADICTED: "REFUTES",
    Label.NOT_ENOUGH_INFORMATION: "NOT ENOUGH INFO",
}
_PTB = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}",
        "-COLON-": ":", "``": '"', "''": '"'}


def normalize_wiki_text(text: str) -> str:
    """Undo FEVER/PTB escaping (-LRB- etc.) and underscores in titles."""
    for k, v in _PTB.items():
        text = text.replace(k, v)
    text = text.replace("_", " ")
    text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return " ".join(text.split())


def parse_lines(lines_field: str) -> dict[int, str]:
    """Parse the dump's ``lines`` field: ``"0\\tSentence\\tlink...\\n1\\t..."``."""
    out: dict[int, str] = {}
    for row in lines_field.split("\n"):
        parts = row.split("\t")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].strip():
            out[int(parts[0])] = parts[1]
    return out


def title_candidates(claim: str, max_len: int = 4) -> list[str]:
    """Capitalised n-grams of a claim as dump-style page ids, longest first."""
    words = re.findall(r"[\w'.-]+", claim)
    cands: list[str] = []
    for n in range(max_len, 0, -1):
        for i in range(len(words) - n + 1):
            gram = words[i:i + n]
            if gram[0][:1].isupper() and gram[-1][:1].isupper():
                cands.append("_".join(gram))
    return list(dict.fromkeys(cands))


def load_pages(wiki_dir: Path, wanted: set[str]) -> dict[str, dict[int, str]]:
    """Stream ``wiki-*.jsonl`` and return ``{page_id: {sentence_no: text}}`` for wanted pages."""
    pages: dict[str, dict[int, str]] = {}
    files = sorted(Path(wiki_dir).glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No wiki-*.jsonl files in {wiki_dir}")
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                # cheap pre-filter before JSON parsing
                m = re.match(r'\{"id": "((?:[^"\\]|\\.)*)"', line)
                if m and json.loads(f'"{m.group(1)}"') not in wanted:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("id") in wanted:
                    pages[row["id"]] = parse_lines(row.get("lines", ""))
    return pages


@dataclass
class Pair:
    id: object
    claim: str
    evidence: str
    label: str  # FEVER label name (training target)
    split: str
    evidence_source: str  # "gold" or "nearest_page"
    pages: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def needed_pages(examples: Iterable[FeverExample]) -> set[str]:
    wanted: set[str] = set()
    for ex in examples:
        if ex.label is Label.NOT_ENOUGH_INFORMATION:
            wanted.update(title_candidates(ex.claim))
        elif ex.evidence_sets:
            wanted.update(ev.wiki_page for ev in ex.evidence_sets[0] if ev.wiki_page)
    return wanted


def build_pairs(
    examples: list[FeverExample],
    pages: dict[str, dict[int, str]],
    split: str,
    seed: int = 13,
    max_sentences: int = 5,
) -> tuple[list[Pair], Counter]:
    """Turn labelled examples into pairs. Returns pairs and a counter of drop reasons."""
    rng = random.Random(seed)
    pairs: list[Pair] = []
    dropped: Counter = Counter()
    for ex in examples:
        if ex.label is None:
            dropped["unlabelled"] += 1
            continue
        if ex.label is Label.NOT_ENOUGH_INFORMATION:
            page = next((t for t in title_candidates(ex.claim) if pages.get(t)), None)
            if page is None:
                dropped["nei_no_matching_page"] += 1
                continue
            sents = sorted(pages[page])
            chosen = sorted(rng.sample(sents, min(2, len(sents))))
            text = " ".join(pages[page][i] for i in chosen)
            evidence = f"{normalize_wiki_text(page)}: {normalize_wiki_text(text)}"
            pairs.append(Pair(ex.id, ex.claim, evidence, FEVER_LABEL_NAMES[ex.label], split,
                              "nearest_page", [page]))
            continue
        if not ex.evidence_sets:
            dropped["no_evidence_annotation"] += 1
            continue
        parts, used = [], []
        missing = False
        for ev in ex.evidence_sets[0][:max_sentences]:
            sent = pages.get(ev.wiki_page or "", {}).get(ev.sentence_id if ev.sentence_id is not None else -1)
            if sent is None:
                missing = True
                break
            parts.append(f"{normalize_wiki_text(ev.wiki_page)}: {normalize_wiki_text(sent)}")
            used.append(ev.wiki_page)
        if missing or not parts:
            dropped["evidence_sentence_not_in_dump"] += 1
            continue
        pairs.append(Pair(ex.id, ex.claim, " ".join(parts), FEVER_LABEL_NAMES[ex.label], split,
                          "gold", list(dict.fromkeys(used))))
    return pairs, dropped


def balanced_subset(examples: list[FeverExample], max_per_class: int | None, seed: int) -> list[FeverExample]:
    """Seeded subset with at most ``max_per_class`` labelled examples per class."""
    labelled = [e for e in examples if e.label is not None]
    if max_per_class is None:
        return labelled
    rng = random.Random(seed)
    out: list[FeverExample] = []
    for label in Label:
        group = [e for e in labelled if e.label is label]
        out += rng.sample(group, min(max_per_class, len(group)))
    rng.shuffle(out)
    return out


def split_train_validation(
    examples: list[FeverExample], val_fraction: float, seed: int
) -> tuple[list[FeverExample], list[FeverExample]]:
    """Hold out a validation slice of the official *train* split, grouped by claim text.

    Used for model selection and threshold tuning only; the official dev/test files stay
    untouched for final evaluation. Identical claim texts never cross the boundary.
    """
    keys = sorted({claim_key(e.claim) for e in examples})
    rng = random.Random(seed)
    rng.shuffle(keys)
    val_keys = set(keys[: int(round(len(keys) * val_fraction))])
    train = [e for e in examples if claim_key(e.claim) not in val_keys]
    val = [e for e in examples if claim_key(e.claim) in val_keys]
    return train, val


def claim_key(claim: str) -> str:
    """Normalised claim text used for leakage checks."""
    return " ".join(re.findall(r"\w+", claim.lower()))


def class_counts(items: Iterable) -> dict[str, int]:
    c: Counter = Counter()
    for it in items:
        lab = it.label
        c[lab.value if isinstance(lab, Label) else str(lab)] += 1
    return dict(c)


def read_pairs(path: Path) -> list[Pair]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(Pair(**json.loads(line)))
    return out
