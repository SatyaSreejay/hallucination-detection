"""FEVER pair construction, splitting and prepare_fever on synthetic fixtures."""

import json

from src.datasets.fever_loader import load_fever_jsonl
from src.datasets.fever_pairs import (
    build_pairs,
    claim_key,
    load_pages,
    needed_pages,
    normalize_wiki_text,
    parse_lines,
    split_train_validation,
    title_candidates,
)
from tests.conftest import FIXTURES

WIKI = FIXTURES / "fever_wiki_synthetic"


def test_text_normalisation_and_line_parsing():
    assert normalize_wiki_text("Homeland_-LRB-TV_series-RRB-") == "Homeland (TV series)"
    assert normalize_wiki_text("founded in 1850 -LRB- by settlers -RRB- .") == "founded in 1850 (by settlers)."
    assert parse_lines("0\tA .\tlink\n1\t\n2\tB .") == {0: "A .", 2: "B ."}
    assert title_candidates("Synthetic Town has a famous bakery.")[0] == "Synthetic_Town"


def test_build_pairs_gold_and_nearest_page():
    examples, _ = load_fever_jsonl(FIXTURES / "fever_synthetic.jsonl")
    pages = load_pages(WIKI, needed_pages(examples))
    assert set(pages) == {"Synthetic_Town", "Example_Country"}  # unrelated page never loaded
    pairs, dropped = build_pairs(examples, pages, "train", seed=1)
    by_id = {p.id: p for p in pairs}
    sup = by_id[1]
    assert sup.label == "SUPPORTS" and sup.evidence_source == "gold"
    assert sup.evidence == "Synthetic Town: Synthetic Town is a town in Example Country."
    ref = by_id[2]
    assert ref.label == "REFUTES"
    assert "founded in 1850 (by settlers)" in ref.evidence and "Its capital is Sample City" in ref.evidence
    nei = by_id[3]
    assert nei.label == "NOT ENOUGH INFO" and nei.evidence_source == "nearest_page"
    assert nei.pages == ["Synthetic_Town"]
    assert dropped["unlabelled"] == 2  # id 6 (no label) and id 7 (unknown label)
    assert dropped["evidence_sentence_not_in_dump"] == 1  # id 9 references a page not in the dump
    for p in pairs:  # the label never leaks into the model input
        assert p.label not in p.evidence and p.label not in p.claim


def test_split_by_claim_text_prevents_leakage():
    examples, _ = load_fever_jsonl(FIXTURES / "fever_synthetic.jsonl")
    train, val = split_train_validation(examples * 3, 0.5, seed=0)
    assert {claim_key(e.claim) for e in train}.isdisjoint({claim_key(e.claim) for e in val})


def test_prepare_fever_script(tmp_path):
    from training.prepare_fever import main

    fever = tmp_path / "fever"
    fever.mkdir()
    (fever / "train.jsonl").write_text((FIXTURES / "fever_synthetic.jsonl").read_text())
    out = tmp_path / "prepared"
    # without the wiki dump only statistics are written
    assert main(["--fever-dir", str(fever), "--wiki-dir", str(tmp_path / "none"), "--out", str(out)]) == 1
    report = json.loads((out / "prepare_report.json").read_text())
    assert report["pairs_built"] is False and report["files"]["train"]["rows"] == 6
    assert "No official labelled held-out split" in report["warning"]
    # with the dump, pairs are written
    assert main(["--fever-dir", str(fever), "--wiki-dir", str(WIKI), "--out", str(out),
                 "--val-fraction", "0.0"]) == 0
    report = json.loads((out / "prepare_report.json").read_text())
    assert report["pairs_built"] and report["splits"]["train"]["pairs"] == 3
    assert len((out / "train.jsonl").read_text().splitlines()) == 3
