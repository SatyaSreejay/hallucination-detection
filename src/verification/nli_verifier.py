"""NLI-based claim verifier (pretrained checkpoint or a checkpoint fine-tuned on FEVER).

Each evidence passage is the *premise* and the claim is the *hypothesis*.

Label mapping (evaluated on FEVER by ``training/evaluate_verifier.py``, not assumed exact):

    entailment    -> SUPPORTED
    contradiction -> CONTRADICTED
    neutral       -> NOT_ENOUGH_INFORMATION (shown as "Insufficient Evidence")

Checkpoints fine-tuned by ``training/train_verifier.py`` use FEVER label names
(SUPPORTS / REFUTES / NOT ENOUGH INFO) and are mapped the same way.

Per-passage decision: the argmax class. Optional minimum scores (``support_threshold``,
``contradiction_threshold``) are only meant to be set from a threshold search on
validation data (``evaluate_verifier.py --tune``); by default none are used.

Aggregation over passages: if some passages support and others contradict the claim, the
evidence conflicts and the verdict is NOT_ENOUGH_INFORMATION with ``mixed_evidence``.
Otherwise any supporting passage gives SUPPORTED, any contradicting passage gives
CONTRADICTED, and neither gives NOT_ENOUGH_INFORMATION. Neutral never means "false".

Scores are softmax outputs: uncalibrated, not probabilities that the claim is true.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from collections.abc import Callable
from pathlib import Path

from src.schemas import EvidencePassage, Label, Verdict
from src.verification.base import VerifierUnavailableError, format_premise

logger = logging.getLogger(__name__)

# (premises, hypotheses) -> one {"entailment", "neutral", "contradiction"} dict per pair
PredictFn = Callable[[list[str], list[str]], list[dict[str, float]]]

SCORE_NOTE = (
    "Softmax outputs of the NLI model for each evidence passage. They are uncalibrated: "
    "useful for comparing passages, not probabilities that the claim is true."
)
NLI_TO_LABEL = {
    "entailment": Label.SUPPORTED,
    "contradiction": Label.CONTRADICTED,
    "neutral": Label.NOT_ENOUGH_INFORMATION,
}


def nli_dependencies_available() -> bool:
    """True if ``transformers`` and ``torch`` are importable (weights not checked)."""
    return all(importlib.util.find_spec(m) is not None for m in ("transformers", "torch"))


def normalise_label_name(name: str) -> str | None:
    """Map NLI or FEVER label names to entailment / contradiction / neutral."""
    name = name.lower().replace("_", " ").strip()
    if name.startswith(("entail", "support")):
        return "entailment"
    if name.startswith(("contra", "refute")):
        return "contradiction"
    if name.startswith(("neutral", "not enough", "nei", "insufficient")):
        return "neutral"
    return None


def label_from_scores(
    scores: dict[str, float],
    support_threshold: float | None = None,
    contradiction_threshold: float | None = None,
) -> Label:
    """Argmax mapping NLI -> application label, with optional validated minimum scores."""
    best = max(("entailment", "contradiction", "neutral"), key=lambda k: scores.get(k, 0.0))
    if best == "entailment" and support_threshold is not None and scores["entailment"] < support_threshold:
        return Label.NOT_ENOUGH_INFORMATION
    if (best == "contradiction" and contradiction_threshold is not None
            and scores["contradiction"] < contradiction_threshold):
        return Label.NOT_ENOUGH_INFORMATION
    return NLI_TO_LABEL[best]


def read_training_record(model_name: str) -> dict | None:
    """The ``training_record.json`` written by train_verifier.py, if ``model_name`` is one."""
    failed = Path(model_name) / "training_record.failed.json"
    path = Path(model_name) / "training_record.json"
    if failed.is_file() and not path.is_file():
        return {"status": "failed"}
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return None


class NLIVerifier:
    """Claim verification with a local NLI sequence-classification model (CPU or GPU)."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
        support_threshold: float | None = None,
        contradiction_threshold: float | None = None,
        cache_dir: Path | None = None,
        predict_fn: PredictFn | None = None,
        allow_download: bool = False,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.support_threshold = support_threshold
        self.contradiction_threshold = contradiction_threshold
        self.cache_dir = cache_dir
        self.allow_download = allow_download
        self.max_length = max_length
        self._predict_fn = predict_fn
        self.training_record = read_training_record(model_name)

    @property
    def method(self) -> str:
        if self.training_record and self.training_record.get("status") == "failed":
            return f"NLI checkpoint from a FAILED training run ({self.model_name}); results are not trustworthy"
        if self.training_record:
            base = self.training_record.get("base_model", "?")
            return f"NLI classifier fine-tuned on FEVER by this project (from {base}; {self.model_name})"
        return f"NLI classifier ({self.model_name}; pretrained, not trained by this project)"

    # --------------------------------------------------------------- loading
    def load(self) -> None:
        """Load tokenizer and model; uses only local files unless downloads are allowed."""
        if self._predict_fn is not None:
            return
        if not nli_dependencies_available():
            raise VerifierUnavailableError(
                "NLI verifier needs `transformers` and `torch` (pip install -r requirements-nli.txt)."
            )
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            kwargs: dict = {"local_files_only": not self.allow_download}
            if self.cache_dir and not Path(self.model_name).is_dir():
                kwargs["cache_dir"] = str(self.cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, **kwargs)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device).eval()
        except Exception as exc:  # noqa: BLE001 - surface any load failure clearly
            hint = (
                " If it is a Hugging Face model id, download it once with "
                "`python -m src.cli download-nli`."
                if not self.allow_download
                else ""
            )
            raise VerifierUnavailableError(
                f"Could not load NLI model '{self.model_name}': {type(exc).__name__}: "
                f"{str(exc)[:200]}.{hint}"
            ) from exc

        id2label = {int(i): normalise_label_name(str(n)) for i, n in model.config.id2label.items()}
        if set(id2label.values()) != {"entailment", "neutral", "contradiction"}:
            raise VerifierUnavailableError(
                f"Model '{self.model_name}' does not expose entailment/neutral/contradiction "
                f"(or SUPPORTS/REFUTES/NOT ENOUGH INFO) labels; got {model.config.id2label}."
            )
        max_length = min(self.max_length, getattr(tokenizer, "model_max_length", self.max_length) or self.max_length)

        def predict(premises: list[str], hypotheses: list[str]) -> list[dict[str, float]]:
            with torch.no_grad():
                enc = tokenizer(premises, hypotheses, truncation="only_first", max_length=max_length,
                                padding=True, return_tensors="pt").to(device)
                probs = torch.softmax(model(**enc).logits, dim=-1).tolist()
            return [{id2label[i]: float(p) for i, p in enumerate(row)} for row in probs]

        self._predict_fn = predict
        logger.info("Loaded NLI model %s on %s", self.model_name, device)

    def score_pairs(
        self, premises: list[str], hypotheses: list[str], batch_size: int = 16
    ) -> list[dict[str, float]]:
        """Batch scoring for evaluation scripts."""
        if self._predict_fn is None:
            self.load()
        assert self._predict_fn is not None
        out: list[dict[str, float]] = []
        for i in range(0, len(premises), batch_size):
            out += self._predict_fn(premises[i:i + batch_size], hypotheses[i:i + batch_size])
        return out

    # ---------------------------------------------------------------- verify
    def verify(self, claim_text: str, passages: list[EvidencePassage]) -> Verdict:
        if not passages:
            return Verdict(Label.NOT_ENOUGH_INFORMATION, self.method, "No evidence passages to compare.")
        try:
            scores = self.score_pairs([format_premise(p) for p in passages], [claim_text] * len(passages))
        except VerifierUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            return Verdict(
                Label.NOT_ENOUGH_INFORMATION,
                self.method,
                "The NLI model failed on this input, so no verdict could be made.",
                error=f"{type(exc).__name__}: {exc}"[:300],
            )
        return self.aggregate(passages, scores)

    def aggregate(self, passages: list[EvidencePassage], scores: list[dict[str, float]]) -> Verdict:
        """Combine per-passage NLI scores into one verdict (see module docstring)."""
        per_passage = []
        supporting: list[tuple[float, EvidencePassage]] = []
        contradicting: list[tuple[float, EvidencePassage]] = []
        for passage, s in zip(passages, scores, strict=True):
            label = label_from_scores(s, self.support_threshold, self.contradiction_threshold)
            per_passage.append({
                "title": passage.title,
                "passage_index": passage.passage_index,
                "entailment": round(s.get("entailment", 0.0), 4),
                "neutral": round(s.get("neutral", 0.0), 4),
                "contradiction": round(s.get("contradiction", 0.0), 4),
                "passage_label": label.value,
            })
            if label is Label.SUPPORTED:
                supporting.append((s["entailment"], passage))
            elif label is Label.CONTRADICTED:
                contradicting.append((s["contradiction"], passage))
        raw = {
            "per_passage": per_passage,
            "decision_rule": "argmax per passage"
            + (f"; min entailment {self.support_threshold}" if self.support_threshold is not None else "")
            + (f"; min contradiction {self.contradiction_threshold}"
               if self.contradiction_threshold is not None else ""),
        }
        supporting.sort(key=lambda x: -x[0])
        contradicting.sort(key=lambda x: -x[0])

        if supporting and contradicting:
            return Verdict(
                Label.NOT_ENOUGH_INFORMATION, self.method,
                f"Evidence conflicts: {len(supporting)} passage(s) entail the claim and "
                f"{len(contradicting)} contradict it, so no verdict is forced.",
                [supporting[0][1], contradicting[0][1]], raw, SCORE_NOTE, mixed_evidence=True,
            )
        if supporting:
            score, best = supporting[0]
            return Verdict(
                Label.SUPPORTED, self.method,
                f"The model judges that the passage from '{best.title}' entails the claim "
                f"(entailment score {score:.2f}).",
                [p for _, p in supporting], raw, SCORE_NOTE,
            )
        if contradicting:
            score, best = contradicting[0]
            return Verdict(
                Label.CONTRADICTED, self.method,
                f"The model judges that the passage from '{best.title}' contradicts the claim "
                f"(contradiction score {score:.2f}).",
                [p for _, p in contradicting], raw, SCORE_NOTE,
            )
        return Verdict(
            Label.NOT_ENOUGH_INFORMATION, self.method,
            "No passage was judged to entail or contradict the claim (all neutral). The "
            "evidence may be off-topic or cover only part of the claim; this does not mean "
            "the claim is false.",
            list(passages[:1]), raw, SCORE_NOTE,
        )
