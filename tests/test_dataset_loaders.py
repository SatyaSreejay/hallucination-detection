"""FEVER and RAGTruth loaders on clearly labelled synthetic fixtures."""

import pytest

from src.datasets.fever_loader import load_fever_jsonl, normalize_fever_label
from src.datasets.ragtruth_loader import context_to_text, load_ragtruth
from src.schemas import Label
from tests.conftest import FIXTURES


@pytest.mark.parametrize("raw,expected", [
    ("SUPPORTS", Label.SUPPORTED), ("REFUTES", Label.CONTRADICTED),
    ("NOT ENOUGH INFO", Label.NOT_ENOUGH_INFORMATION), ("not enough info", Label.NOT_ENOUGH_INFORMATION),
    ("NOT_ENOUGH_INFO", Label.NOT_ENOUGH_INFORMATION), ("MAYBE", None), (None, None), (3, None),
])
def test_fever_label_normalization(raw, expected):
    assert normalize_fever_label(raw) is expected


def test_fever_loader_handles_malformed_rows_and_preserves_evidence():
    examples, report = load_fever_jsonl(FIXTURES / "fever_synthetic.jsonl")
    assert [e.id for e in examples] == [1, 2, 3, 6, 7, 9]
    assert report.n_malformed == 3  # bad JSON, missing claim, non-object row
    assert report.n_unlabelled == 1 and report.unknown_labels == {"MAYBE": 1}
    assert report.errors and report.split == "fever_synthetic"
    by_id = {e.id: e for e in examples}
    assert by_id[2].label is Label.CONTRADICTED and by_id[2].original_label == "REFUTES"
    assert len(by_id[2].evidence_sets) == 2 and len(by_id[2].evidence_sets[0]) == 2
    ev = by_id[2].evidence_sets[0][0]
    assert (ev.annotation_id, ev.evidence_id, ev.wiki_page, ev.sentence_id) == (101, 201, "Synthetic_Town", 2)
    nei = by_id[3].evidence_sets[0][0]
    assert nei.wiki_page is None and nei.sentence_id is None
    assert by_id[6].label is None and by_id[7].label is None
    assert by_id[9].evidence_sets[0][0].wiki_page == "Synthetic_Page"  # flattened format


def test_fever_loader_limit():
    examples, _ = load_fever_jsonl(FIXTURES / "fever_synthetic.jsonl", limit=2)
    assert len(examples) == 2


def test_ragtruth_schema_detection_and_annotation_preservation():
    examples, schema = load_ragtruth(FIXTURES / "ragtruth_synthetic" / "response.jsonl")
    assert schema.response_text_key == "response" and schema.labels_key == "labels"
    assert schema.context_key == "source_info"
    assert {"start", "end", "label_type", "meta"} <= set(schema.span_keys)
    assert schema.label_types == {"Evident Conflict": 1, "Subtle Baseless Info": 1}
    assert schema.n_malformed == 2  # broken JSON line + non-string response
    assert len(examples) == 3
    r1 = examples[0]
    assert r1.is_hallucinated and r1.split == "test" and r1.task_type == "QA"
    span = r1.spans[0]
    assert r1.response[span.start:span.end] == span.text == "has 5000 residents"
    assert span.is_conflict and span.extra["meta"] == "context says 3000"
    assert span.extra["implicit_true"] is False
    assert "3000 residents" in r1.context and "Question:" in r1.context
    assert not examples[1].is_hallucinated
    assert examples[2].context.startswith("A short synthetic news article")
    assert not examples[2].spans[0].is_conflict


def test_ragtruth_missing_source_file_and_unknown_schema(tmp_path):
    f = tmp_path / "response.jsonl"
    f.write_text('{"id": 1, "output": "Text here.", "hallucination_labels": [{"start": 0, "end": 4, "type": "x"}]}\n')
    examples, schema = load_ragtruth(f)
    assert schema.response_text_key == "output" and schema.labels_key == "hallucination_labels"
    assert schema.source_file is None and schema.n_missing_context == 1
    assert examples[0].spans[0].label_type == "x"
    g = tmp_path / "weird.jsonl"
    g.write_text('{"id": 1, "foo": "bar"}\n')
    examples, schema = load_ragtruth(g)
    assert examples == [] and "No response text field" in schema.errors[-1]


def test_context_to_text_variants():
    assert context_to_text("abc") == "abc"
    assert context_to_text(None) == ""
    assert "Question: q" in context_to_text({"question": "q", "passages": ["p1", "p2"]})
    assert '"name"' in context_to_text({"name": "Cafe", "rating": 4})


def test_ragtruth_sentence_labels_and_implicit_true(tmp_path):
    from src.datasets.ragtruth_loader import dataset_statistics, gold_spans, sentence_labels

    examples, _ = load_ragtruth(FIXTURES / "ragtruth_synthetic" / "response.jsonl")
    labels = sentence_labels(examples[0])
    assert [(s.text, g) for s, g in labels] == [
        ("Synthetic Town was founded in 1850.", False), ("It has 5000 residents.", True)]
    f = tmp_path / "response.jsonl"
    f.write_text('{"id": 1, "response": "Paris is big. Rome is old.", "split": "test", '
                 '"labels": [{"start": 0, "end": 13, "text": "Paris is big.", "label_type": "Evident Baseless Info", '
                 '"implicit_true": true}]}\n')
    ex, _ = load_ragtruth(f)
    assert gold_spans(ex[0]) == [] and len(gold_spans(ex[0], include_implicit_true=True)) == 1
    assert [g for _, g in sentence_labels(ex[0])] == [False, False]
    assert [g for _, g in sentence_labels(ex[0], include_implicit_true=True)] == [True, False]
    stats = dataset_statistics(examples)
    assert stats["responses"] == 3 and stats["responses_with_any_span"] == 2
