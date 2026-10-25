# Hallucination Detection Using Machine Learning and Evidence Verification

A locally runnable research prototype. A user asks a question, a pretrained Llama model answers it, and a **separate** verification system checks each claim in the answer against Wikipedia evidence. Each claim is labelled **Supported**, **Contradicted** or **Insufficient Evidence**. If no verdict can be made, the claim is labelled **Retrieval Failure** or **Verification Error** instead.

This is evidence-based verification. It can flag likely hallucinations, but it is not a new LLM, and it cannot guarantee that any statement is correct.

> **Results policy.** No metrics ship with this repository. Every number in the dashboard is read from `results/*.json` files written by real runs. Unit-test fixtures are synthetic and are never results. See [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) for what has and has not been run.

## Pipeline

```
User question
 → LLM answer: Groq API (Llama 3.x, default) or Ollama   src/llm/, src/pipeline.py
 → spaCy Sentencizer: candidate claims S1, S2, ...        src/claims/extractor.py
 → Wikipedia search + article text, passage ranking       src/retrieval/wikipedia.py, passages.py
 → (evidence, claim) pairs → NLI classifier               src/verification/nli_verifier.py
 → aggregation over passages → verdict + evidence + scores
 → Streamlit dashboard                                    app.py
```

Each stage is a separate module with its own tests. The verifier only ever receives the claim text and the evidence passages.

| Module | Responsibility |
| --- | --- |
| `src/config.py` | All settings (environment variables or `.env`) |
| `src/schemas.py` | `Answer` (frozen), `Sentence`, `Claim`, `EvidencePassage`, `RetrievalResult`, `Verdict`, `Outcome`, `Summary` |
| `src/llm/` | Generation backends behind one `LLMClient` interface: `api_client.py` (OpenAI-compatible: Groq, OpenRouter, OpenAI, custom), `ollama_client.py` (local), `factory.py` (picks one from `LLM_PROVIDER`). Typed errors for auth, rate limit, timeout, missing model and malformed responses |
| `src/claims/extractor.py` | spaCy sentence extraction with ids, skip reasons and multi-assertion flags; optional LLM decomposition |
| `src/retrieval/` | MediaWiki search plus full article text (TextExtracts), passage chunking and BM25-style ranking, caching and retries; a context retriever for RAGTruth |
| `src/verification/` | NLI verifier (pretrained or fine-tuned), LLM-judgment fallback, and verifier selection |
| `src/experiments/injection.py` | Controlled Hallucination Test (side-by-side original and test copy) |
| `src/experiments/mutation.py` | Batch controlled-mutation experiment with validation |
| `src/datasets/` | FEVER loader, FEVER pair builder (wiki dump), RAGTruth loader and sentence-label conversion |
| `src/evaluation.py`, `src/dataset_eval.py` | Metrics (scikit-learn) and dataset evaluation runs |
| `training/` | `prepare_fever.py`, `train_verifier.py`, `evaluate_verifier.py` |
| `src/cli.py` | Command-line access to everything |

## Install

Python 3.10 or newer. GPU optional.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                      # PowerShell: Copy-Item .env.example .env
```

Put a contact in `WIKI_USER_AGENT` in `.env`, as Wikimedia's API policy asks.

spaCy needs no model download, because the app uses a blank English pipeline with the rule-based Sentencizer.

### Answer generation: Groq API (default)

The default backend is Groq's free tier, which serves open-weight models over an OpenAI-compatible API. Nothing is downloaded.

1. Create a free key at https://console.groq.com/keys.
2. Put it in `.env` (never in the code, and never commit `.env`):

```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
LLM_MODEL=openai/gpt-oss-20b    # must be one of the models `status` lists for your key
```

3. Check it: `python -m src.cli status` shows `"generation": {"provider": "groq", "model_available": true}`.

Notes:

* The free tier has per-minute and per-day request limits. The client retries on HTTP 429 (honouring `Retry-After`) and then reports a rate-limit error; nothing is substituted.
* Groq's model list depends on the account and changes over time; in October 2026 this project's key listed no Llama chat models, so the default is `openai/gpt-oss-20b` (open-weight, 20B). That is not Llama 3.2 3B: for a Llama answer generator use `LLM_PROVIDER=openrouter` with a `:free` Llama model, or `ollama`. Record the provider and model with any results; both are stored in each answer (`provider`, `model`).
* The key is read from the environment only. It is not shown in the dashboard, logs or error messages.
* Other backends: `LLM_PROVIDER=openrouter` (`OPENROUTER_API_KEY`, free `:free` models), `openai` (`OPENAI_API_KEY`, paid), `custom` (any OpenAI-compatible endpoint: set `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`), or `ollama` (below). The dashboard sidebar also has a backend selector.

### Optional: Llama 3.2 via Ollama (local; the app never pulls models)

Set `LLM_PROVIDER=ollama`, then:

1. Install Ollama from https://ollama.com/download.
2. Start and prepare it:

```bash
ollama serve                 # or start the Ollama desktop app
ollama pull llama3.2         # Llama 3.2 3B, about 2 GB
ollama list
```

* If local inference is too slow, use `ollama pull llama3.2:1b` and set `OLLAMA_MODEL=llama3.2:1b`.
* You can also point `OLLAMA_HOST` at an Ollama server on a stronger machine.

### NLI verifier

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU build (~200 MB); skip if you have CUDA torch
pip install -r requirements-nli.txt
python -m src.cli download-nli        # cross-encoder/nli-deberta-v3-xsmall, about 0.3 GB, into models/hf-cache
```

If the NLI model is not available, `VERIFIER=auto` falls back to an LLM-judgment verifier. The UI labels that fallback as **not a trained classifier**.

## Run

```bash
python -m src.cli status       # generation backend, Wikipedia and NLI availability
streamlit run app.py           # http://localhost:8501
```

The dashboard has four tabs:

* **Overview** shows service status and exactly which verifier is active.
* **Ask & verify** has the question box, the **Generate Answer** button, the unmodified original answer, and a claim analysis card for each claim. A card shows the id, the exact text, the outcome, the retrieved passages with Wikipedia title and link, the per-passage model scores, and the verifier's explanation, which is kept separate from the source text. The optional **Controlled Hallucination Test** is below the claim cards.
* **Evaluation** shows saved results by dataset (FEVER, RAGTruth, controlled) and a baseline vs fine-tuned comparison. It also runs FEVER end-to-end, RAGTruth and batch-mutation evaluations.
* **Datasets** shows file status, the dataset report and the FEVER preparation report.

CLI equivalents:

```bash
python -m src.cli ask "When was the Eiffel Tower built?"
python -m src.cli controlled-test "Tell me about malaria." --example malaria
python -m src.cli dataset-report
```

## Claim extraction (spaCy)

* The answer is split with spaCy's rule-based Sentencizer, line by line so that list items do not merge.
* Every sentence gets an id (`S1`, `S2`, …), its exact text and its character span, in the original order.
* Questions, greetings, opinions, advice and fragments under 4 words are shown but not verified, and their reason is displayed.
* **Sentence splitting is not atomic-fact extraction.** Sentences that look like they contain several assertions are **flagged**, not split. The signals are semicolons, coordinated or relative clauses, parentheticals, and two or more numbers or dates. A flagged sentence is verified as a whole, so it can only be Supported if the evidence supports all of it.
* Optionally, set `CLAIM_EXTRACTOR=llm` to have the generation LLM decompose sentences into sub-claims (`S2.1`, `S2.2`). The JSON output is validated, and sub-claims that cannot be traced back to a sentence are dropped. If decomposition fails, the app falls back to spaCy sentences and shows a visible warning.

## Evidence retrieval (Wikipedia)

1. The answer's subject is found first: the top Wikipedia hit for the question (or for the first claim when an answer is pasted). Claims that do not name it ("It was designated a UNESCO site in 1983") are searched as "subject: claim". The subject only steers search; the verifier never sees it.
2. For each claim, the MediaWiki search API (`list=search`) finds the top pages. If the full sentence returns nothing, a keyword query is tried.
3. Passages from pages about other subjects (no content word of the page title appears in the claim or the answer's subject) are not used for the verdict. Without this, a claim about the Taj Mahal's UNESCO listing was "contradicted" by the Machu Picchu article.
4. The **full article text** is fetched with `prop=extracts&explaintext` (TextExtracts), along with the canonical URL. This gives clean plain text, which is easier to use than parsing HTML from `action=parse`.
5. Articles are cut into passages and ranked by lexical overlap with the claim. The top passages (3 by default) go to the verifier.
6. Requests have timeouts, retries with backoff, a minimum interval between calls, and an in-memory and on-disk cache.
7. Outcomes:
   * No results means Insufficient Evidence. Absence of evidence is never treated as falsity.
   * A network or API error means Retrieval Failure.
   * Retrieved text is untrusted data. It is shown verbatim and never followed as instructions.

No vector database is used. Lexical ranking was enough for the first version.

## Verification

* The model input is a text pair: **(evidence passage, claim)**, with the evidence as the premise and the claim as the hypothesis. The same format is used in training, evaluation and the app.
* The page title is prefixed to the passage so that pronouns can be resolved.
* **Label mapping**, evaluated on FEVER rather than assumed to be exact:

  | NLI class | App label |
  | --- | --- |
  | entailment | Supported |
  | contradiction | Contradicted |
  | neutral | Insufficient Evidence |

  Neutral means the evidence does not settle the claim. It never means the claim is false. Fine-tuned checkpoints use the FEVER names SUPPORTS, REFUTES and NOT ENOUGH INFO, and are mapped the same way.
* **Per passage:** the argmax class. Minimum-score thresholds are **off by default**. They can be applied only from a file produced by `evaluate_verifier.py --tune` on the validation split, which is selectable in the sidebar.
* **Across passages:**
  * If some passages support the claim and others contradict it, the verdict is Insufficient Evidence, flagged as **mixed evidence**. No verdict is forced.
  * Otherwise, any supporting passage gives Supported, then any contradicting passage gives Contradicted, and if neither applies the verdict is Insufficient Evidence.
* **Scores:** each passage's softmax scores are shown. They are uncalibrated and are never presented as the probability that a claim is true. The summary's "contradicted share" is descriptive, and its denominator is stated.

## Controlled Hallucination Test

This is an optional mode under the claim analysis.

1. The original Llama answer is a frozen object. It is never edited.
2. A **separate test copy** is created in one of two ways:
   * **Insert a reviewed false claim** from `data/controlled_examples.json`. These are 8 true/false pairs with clear contradictions, each with its Wikipedia article. They were written and checked by hand, so re-check them before relying on them.
   * **Edit one sentence**: pick a sentence, optionally start from a rule-based date, number or negation suggestion, and review or edit the false version yourself. Random entity swaps are not offered, because they produce nonsensical sentences.
3. The record keeps the original text, the modified text, the changed span, the transformation and its origin.
4. Both answers run through the same extraction and verification pipeline. The UI shows them side by side with the injected claim highlighted, and labels the copy *"CONTROLLED TEST COPY … Not model output"*.
5. The UI reports the verifier's actual outcome for the injected claim, including when it is **not** detected.

A batch variant (Experiment D) is in Evaluation → Controlled mutations. Each mutation is validated against reference evidence, and the detector sees only the claim text.

## Datasets

### FEVER (primary; train and evaluate the verifier)

FEVER claims reference evidence as `(Wikipedia page, sentence number)` in a **2017 Wikipedia dump**. The labelled files do not contain evidence text, so you need these files:

| File | Purpose |
| --- | --- |
| `data/fever/train.jsonl` | Training split |
| `data/fever/shared_task_dev.jsonl` (and/or `paper_dev.jsonl` / `paper_test.jsonl`) | Labelled held-out evaluation. `shared_task_test` has no labels. |
| `data/fever/wiki-pages/wiki-*.jsonl` | From `wiki-pages.zip`, about 1.7 GB unzipped. Needed to build evidence text. |

All three come from https://fever.ai/dataset/fever.html; the code is at https://github.com/awslabs/fever.

```bash
python -m training.prepare_fever --max-per-class 3000 --max-per-class-eval 1000 --val-fraction 0.1
```

`prepare_fever` does the following:

* **Splits:**
  * The official train split is used for training.
  * A validation slice is held out *from train*, grouped by claim text so that identical claims never cross the boundary. It is used only for model selection and threshold tuning.
  * The official dev/test files are used only for final evaluation. Any dev claim whose text also appears in train is removed.
* **Subsets:** class-balanced and seeded, so runs fit in a one-day budget. Real counts per split and class are written to `data/fever/prepared/prepare_report.json`.
* **Evidence:**
  * SUPPORTS and REFUTES use the sentences of the first annotated evidence set.
  * NOT ENOUGH INFO has no annotated evidence. Its evidence is a sample of sentences from the page whose title matches a capitalised n-gram of the claim (a nearest-page heuristic). Claims without a matching page are dropped and counted.
* **Labels:** SUPPORTS → Supported, REFUTES → Contradicted, NOT ENOUGH INFO → Insufficient Evidence. Labels are targets only, never part of the input.

### Experiments A and B (FEVER, annotated-evidence setting)

```bash
# A: pretrained baseline (download once with --allow-download, ~0.3 GB)
python -m training.evaluate_verifier --model cross-encoder/nli-deberta-v3-xsmall --split val --tune
python -m training.evaluate_verifier --model cross-encoder/nli-deberta-v3-xsmall --split shared_task_dev
python -m training.evaluate_verifier --model cross-encoder/nli-deberta-v3-xsmall --split shared_task_dev \
    --thresholds results/thresholds_nli_deberta_v3_xsmall.json

# B: fine-tune, then evaluate on the same held-out split
python -m training.train_verifier --base-model microsoft/deberta-v3-base --allow-download \
    --epochs 1 --batch-size 8 --grad-accum 2 --max-length 256 --out models/fever-verifier
python -m training.evaluate_verifier --model models/fever-verifier --split shared_task_dev
```

**Training choices:**

* **Base model:** `microsoft/deberta-v3-base` (MIT licence, about 0.7 GB) gets a new 3-way head. If you start from an NLI checkpoint instead, its entailment/contradiction/neutral head is reused with the correct label order.
* **Compute:** on CPU, deberta-v3-base is slow, roughly a few hours for 9,000 pairs per epoch, and depends heavily on the machine. On a CPU-only machine, use `--base-model cross-encoder/nli-deberta-v3-xsmall` and `--max-train`.
* **Training record:** each run writes `training_record.json` with the arguments, seed, data counts, per-epoch validation metrics, duration and hardware. A failed run writes `training_record.failed.json`, and the app then labels the checkpoint as untrustworthy.
* **Using it in the app:** select `models/fever-verifier` in the sidebar, or set `NLI_MODEL=models/fever-verifier`.

**Interpreting the results:**

* These experiments measure the verifier with annotated evidence. They are not end-to-end scores.
* The **FEVER (end-to-end)** tab measures the whole pipeline with live Wikipedia retrieval. Today's Wikipedia differs from the 2017 dump, so some gold labels will not match current pages.
* Good FEVER scores do not show that the system generalises to arbitrary topics or to free-form LLM answers.

### RAGTruth (supplementary; Experiment C)

Copy `dataset/response.jsonl` and `dataset/source_info.jsonl` from https://github.com/ParticleMedia/RAGTruth (MIT licence) into `data/ragtruth/`.

RAGTruth is a **different task**. It annotates character spans in RAG responses that conflict with, or are not supported by, the *provided context*. Its labels are never mixed with FEVER labels, and it is not used for training.

**How span annotations become sentence labels:**

* The evidence for each response is its own source context, not Wikipedia.
* The response is split with the same spaCy Sentencizer the app uses.
* A sentence is gold *hallucinated* if any annotated span overlaps it. Spans marked `implicit_true` (correct, just not in the context) are excluded by default, with a toggle to include them.
* A response is gold hallucinated if it has at least one counted span.

**Predictions:**

* *Strict:* the sentence was Contradicted.
* *Lenient:* the sentence was not Supported. This is closer to RAGTruth's "baseless info" category, but it also counts retrieval misses.

Sentence-level and response-level metrics are reported for both rules.

**Scope:** QA and Summary tasks only by default, from the `test` split. Data2txt contexts are JSON tables, and NLI over JSON is unreliable.

**Limitations of this conversion:**

* One short span marks the whole sentence as hallucinated.
* Sentences the extractor skips are excluded and counted. In the test split that excludes 21 hallucinated QA/Summary sentences.
* "Baseless info" is not the same as "contradicted".
* Scores are not comparable to span-level RAGTruth baselines.

## Tests

```bash
python -m pytest     # unit tests; the LLM backends, Wikipedia and models are mocked; real network access is blocked
ruff check .
```

## Known limitations

* A sentence is not an atomic claim. Multi-assertion sentences are only flagged.
* Pronoun-initial sentences ("It was built…") are verified with the page title prepended to the evidence, but retrieval can still miss the subject.
* Retrieval is lexical. Wrong pages or passages usually lead to Insufficient Evidence, and occasionally to a spurious Contradicted on a related fact.
* NLI models read at most 512 tokens per pair, and their scores are uncalibrated.
* The LLM fallback verifier with a 3B model is weak.
* Wikipedia can be wrong or out of date. Supported means "consistent with the retrieved Wikipedia text".
* FEVER NOT ENOUGH INFO evidence comes from a heuristic, which affects that class's metrics.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "No API key configured for provider 'groq'" | Put `GROQ_API_KEY=...` in `.env` and restart the app |
| "rejected the API key (HTTP 401)" | The key is wrong or revoked; create a new one in the Groq console |
| "rate limit or quota reached (HTTP 429)" | Free-tier limit; wait a minute, or lower the evaluation sample size |
| "Model '...' is not available on groq" | Set `LLM_MODEL` to a model listed by `python -m src.cli status` |
| "Cannot reach Ollama" (only with `LLM_PROVIDER=ollama`) | Start `ollama serve`, and check `OLLAMA_HOST` |
| "Model 'llama3.2' was not found" | `ollama pull llama3.2`, or pick an installed model in the sidebar |
| Timeouts | Increase `OLLAMA_TIMEOUT`, or use `llama3.2:1b` |
| Every claim shows Retrieval Failure | `en.wikipedia.org` is unreachable (offline or proxy); run `python -m src.cli status` |
| "NLI verifier unavailable" | Install `requirements-nli.txt`, then run `python -m src.cli download-nli` |
| `prepare_fever` says wiki dump missing | Unzip `wiki-pages.zip` into `data/fever/wiki-pages/` |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
