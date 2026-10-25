# Implementation report (revised spec, 2026-10-22)

> **Update 2026-10-23:** all four experiments have now been run on the project laptop. The
> results are in [Experiment results](#experiment-results-run-2026-10-22--23) below. The next two
> sections describe the original build environment and are kept for the record; their "not run"
> and "no metrics" statements are superseded.

## Experiment results (run 2026-10-22 / 23)

### Hardware and setup

* **Machine:** Windows 11 laptop, Intel i5-10300H, 7.8 GB RAM, **NVIDIA GeForce GTX 1650 (4 GB)**.
* **Software:** Python 3.14.6, torch 2.14.1+cu126, transformers 5.20.2.
* **LLM:** Groq API, `openai/gpt-oss-20b`. **Evidence:** live Wikipedia (MediaWiki API).
* **FEVER pairs:** `training/prepare_fever.py` output: 8,888 train / 2,959 val pairs (from official
  train, grouped by claim) and 2,950 `shared_task_dev` pairs. SUPPORTS/REFUTES use annotated gold
  evidence; NOT ENOUGH INFO uses the nearest-page heuristic.

### Experiment A: pretrained NLI baseline on FEVER

`cross-encoder/nli-deberta-v3-base`, not trained by this project.

| Split | Decision rule | n | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| val | thresholds tuned here (support 0.8, contradiction 0.9) | 2,959 | 0.811 | 0.808 |
| shared_task_dev | argmax | 2,950 | 0.802 | 0.797 |
| shared_task_dev | thresholds from val | 2,950 | 0.810 | **0.807** |

The steps list below names `nli-deberta-v3-xsmall` for Experiment A; the `-base` model was used instead.

### Experiment B: fine-tuned on FEVER

* **Run:** `microsoft/deberta-v3-base` with a new 3-way head, all 8,888 training pairs, 1 epoch,
  batch 4 × gradient accumulation 4 (effective 16), max length 128, lr 2e-5, seed 13, mixed precision
  on the GPU. 35.4 min training + 1 min validation. Record: `models/fever-verifier/training_record.json`.
* **Fix found while running:** transformers 5 keeps a checkpoint's stored dtype, so the model loaded in
  fp16 and AMP refused to unscale fp16 gradients. `train_verifier.py` now loads fp32 master weights
  (`dtype=torch.float32`). The same bug made CPU training extremely slow, because it ran fp16 maths on the CPU.
* **Earlier attempt, kept for transparency:** 3,000 pairs × 1 epoch reached val macro F1 0.551 with
  REFUTES F1 0.04: it labelled almost every refuted claim SUPPORTS. That run was replaced by the full-data run.

| Split | n | Accuracy | Macro F1 | SUPPORTS F1 | REFUTES F1 | NEI F1 |
| --- | --- | --- | --- | --- | --- | --- |
| val (model selection) | 2,959 | 0.951 | 0.951 | 0.935 | 0.928 | 0.990 |
| **shared_task_dev** | 2,950 | **0.945** | **0.945** | 0.932 | 0.922 | 0.983 |

**A vs B on shared_task_dev:**

| | A (pretrained, tuned thresholds) | B (fine-tuned, argmax) |
| --- | --- | --- |
| Macro F1 | 0.807 | **0.945** |
| SUPPORTED F1 | 0.907 | 0.932 |
| CONTRADICTED F1 | 0.798 | 0.922 |
| NOT ENOUGH INFO F1 | 0.715 | 0.983 |

**Caveat: the NEI shortcut.** In every prepared split, NOT ENOUGH INFO evidence comes from a different
source (nearest page) and is longer (about 270 vs 200 characters) than the gold evidence used for the
other two classes. The fine-tuned model can partly recognise NEI from the evidence style alone. The
CONTRADICTED gain (+0.12) is the most trustworthy improvement; the NEI gain (+0.27) is inflated.

### Experiment C: RAGTruth transfer

`python -m src.cli eval-ragtruth data/ragtruth/response.jsonl -n 300`: 300 test responses, QA and Summary,
seed 13, argmax, implicit_true excluded, 167 skipped sentences excluded. The label is HALLUCINATED when the
verifier returns Contradicted.

| Level, rule | A: pretrained base | B: fine-tuned |
| --- | --- | --- |
| Sentence, strict (n = 1,665) | **0.559** | 0.495 |
| Sentence, lenient | **0.374** | 0.270 |
| Response, strict (n = 300) | **0.561** | 0.474 |
| Response, lenient | **0.196** | 0.172 |

Only 59 of 1,665 sentences are hallucinated. The baseline found 13 of them (recall 0.22, precision 0.13),
and the fine-tuned model found 3 (recall 0.05, precision 0.03). **Fine-tuning on FEVER did not transfer to
RAGTruth; it hurt.** Likely reasons (not tested separately):

* RAGTruth checks faithfulness to long documents, whereas FEVER checks short claims against short evidence.
* Training truncated evidence at 128 tokens.
* The NEI shortcut makes long, loosely matching evidence look like "not enough info".
* B started from plain DeBERTa, not from an NLI model.

### Experiment D: controlled hallucination test

For each of the 8 reviewed false claims:

* one Groq answer was generated for "Tell me about <topic>.";
* the false claim was inserted into a test copy;
* the **same** test copy was verified with both models (live Wikipedia, argmax).

The detail is in `results/controlled_tests__baseline_vs_finetuned__20261023.json`, which also includes B2
(run on the identical test copies).

| False claim | A: pretrained | B: fine-tuned |
| --- | --- | --- |
| Malaria is caused by bacteria. | Insufficient | **Contradicted** |
| The Eiffel Tower is located in Berlin, Germany. | **Contradicted** | Insufficient |
| Neil Armstrong … walk on the Moon, in 1975. | **Contradicted** | Insufficient |
| Hamlet is a tragedy written by Charles Dickens. | **Contradicted** | **Contradicted** |
| The Pacific Ocean is the smallest ocean on Earth. | **Contradicted** | **Contradicted** |
| Albert Einstein was born in Lisbon, Portugal. | **Contradicted** | Insufficient |
| Penicillin was discovered by Marie Curie in 1928. | **Contradicted** | Insufficient |
| Water boils at 50 degrees Celsius … | Insufficient | **Contradicted** |
| **Detected** | **6 / 8** | 4 / 8 |

* Every miss was "Insufficient Evidence", never "Supported". This matches the NEI bias seen on RAGTruth.
* The two models miss different claims, so they are complementary.
* With n = 8 this is illustrative, not statistically meaningful.
* The verifiers also marked claims in the *unmodified* answers as Contradicted: 8 for A and 5 for B.
  These were not checked by hand, so they may be real LLM errors or false alarms.

### Summary

Fine-tuning gives a large **in-domain** gain on FEVER (dev macro F1 0.807 → 0.945), but the fine-tuned
model generalises **worse** outside FEVER (RAGTruth, and live retrieval in the controlled test). Most likely
it learned FEVER's evidence style, including the NEI shortcut, rather than general faithfulness checking.
For the app, the pretrained baseline remains the better default for open-ended answers. A follow-up
that fine-tuned from the NLI model (B2, below) made this pattern stronger, not weaker.

### Experiment B2: fine-tuning from the NLI model (follow-up)

**Hypothesis:** starting from `cross-encoder/nli-deberta-v3-base` keeps its general NLI skill and
improves transfer.

**Run:** same data and settings as B, but the base model is the NLI cross-encoder, whose
entailment/contradiction/neutral head is reused. 43.7 min training on the GPU. Record:
`models/fever-verifier-nli/training_record.json`.

| | A: pretrained | B: from DeBERTa | B2: from NLI model |
| --- | --- | --- | --- |
| FEVER val macro F1 | 0.808 (tuned) | 0.951 | 0.954 |
| **FEVER shared_task_dev macro F1** | 0.807 | 0.945 | **0.950** |
| dev CONTRADICTED F1 | 0.798 | 0.922 | **0.939** |
| dev NEI F1 | 0.715 | **0.983** | 0.962 |
| **RAGTruth sentence, strict** | **0.559** | 0.495 | 0.485 |
| RAGTruth response, strict | **0.561** | 0.474 | 0.452 |
| RAGTruth hallucinated sentences found (of 59) | **13** | 3 | 0 |
| **Controlled test, detected (of 8)** | **6** | 4 | 1 |

**Result: the hypothesis is not supported.**

* B2 is the best model on FEVER dev, including the best CONTRADICTED F1.
* B2 transfers worst of the three: on RAGTruth (strict) it found no hallucinated sentences.
* In the controlled test it labelled 7 of the 8 injected false claims Insufficient Evidence. It also
  marked no claim in the unmodified answers as Contradicted.

So starting from an NLI model does not stop fine-tuning on this FEVER data from overriding the general
NLI behaviour. The pattern across B and B2 points to the **evidence-style shortcut**. Live Wikipedia
passages and RAGTruth source documents are long and only partly relevant, like the nearest-page evidence
that always meant NEI in training, so the fine-tuned models answer "not enough info".

**What would test this:** rebuild the FEVER pairs so that NEI evidence is drawn the same way as the other
classes (retrieved, similar length), or so that SUPPORTS/REFUTES evidence is also padded with retrieved
context. Then retrain and re-run C and D. Until then, **the pretrained baseline (A) is the recommended
verifier for the app**; B and B2 are best used for FEVER-style claim/evidence pairs.

## Environment the code was built in

* **Container:** cloud, Python 3.13, 4 CPUs, 15 GB RAM, **no GPU**.
* **Not available:** Ollama is not installed, and no Groq API key is configured here.
* **Network policy:**
  * blocks `en.wikipedia.org`, `huggingface.co`, `fever.ai` and `download.pytorch.org`;
  * the old FEVER S3 bucket returns 403;
  * PyPI works.
* **What this means:**
  * no Llama answers, no live Wikipedia retrieval, and no pretrained or fine-tuned NLI model could be run here;
  * the FEVER Wikipedia dump and the dev/test splits could not be obtained.
* **None of the following has been run:**
  * Experiment A (pretrained baseline)
  * Experiment B (fine-tuning)
  * Experiment C (RAGTruth predictions)
  * Experiment D (controlled test with real models)
* **There are no metrics.** The `results/` folder contains only the dataset report.

## Datasets actually processed

### FEVER `train.jsonl` (supplied)

* 145,449 rows, 0 malformed: SUPPORTS 80,035, REFUTES 29,775, NOT ENOUGH INFO 35,639.
* Evidence is given as page/sentence ids only (109,810 claims have ids). No evidence text is included.
* `training/prepare_fever.py` ran on it and wrote `data/fever/prepared/prepare_report.json`:
  * a held-out validation slice of train, grouped by claim text so identical claims never cross the boundary;
  * class-balanced subsets with seed 13: 9,000 training claims (3,000 per class) and 3,000 validation claims (1,000 per class).
* **Pairs were not built**, because the FEVER Wikipedia dump (`wiki-pages.zip`) is required and was not available.
* **No official labelled held-out split** (`shared_task_dev`, `paper_dev` or `paper_test`) is present, so final FEVER evaluation also needs one of those files.

### RAGTruth (supplied zip, MIT licence)

* **Schema:**
  * `response.jsonl` has id, source_id, model, temperature, labels, split, quality, response.
  * `source_info.jsonl` has source_id, task_type, source, source_info, prompt.
  * Each span has start, end, text, label_type, meta, implicit_true, due_to_null.
* **Loaded:** 17,790 responses (15,090 train / 2,700 test) from 6 models across QA, Summary and Data2txt.
* **Spans:** 14,289 spans. All of their character offsets match the response text. 1,928 spans are marked implicit_true.
* **Sentence-level conversion** (spaCy, implicit_true excluded), test split:

  | Task | Hallucinated sentences | Clean sentences |
  | --- | --- | --- |
  | QA | 292 | 5,056 |
  | Summary | 246 | 4,538 |

  Another 21 hallucinated QA/Summary test sentences are skipped by the extractor.
* The full counts are in `results/dataset_report.json` (`python -m src.cli dataset-report`). No hallucination predictions were made, because no verifier model could run here.

## What is implemented

* **Answer generation:** one `LLMClient` interface with two backends, chosen by `LLM_PROVIDER`. The default is **Groq's free tier** (`openai/gpt-oss-20b`; the user's Groq key listed no Llama chat models) through an OpenAI-compatible client that also serves OpenRouter, OpenAI and custom endpoints. Local Ollama (Llama 3.2 3B) remains available. Both map auth, rate-limit (with `Retry-After` retries), timeout, missing-model and malformed-response errors to clear messages. API keys come from `.env` only and never appear in the UI, logs or errors. Each answer records its `provider` and `model`.
* **Claim extraction:**
  * spaCy Sentencizer gives ids S1…Sn in order, with exact spans.
  * Non-factual sentences are shown but skipped, with the reason.
  * Multi-assertion sentences are flagged, not decomposed.
  * Optional Ollama decomposition into sub-claims, with validation and a visible fallback.
* **Wikipedia retrieval:**
  * search plus full article text, with real titles and URLs;
  * passage ranking;
  * request limits, timeouts, retries and caching;
  * explicit "no results" versus "retrieval failure".
* **Verification:**
  * NLI over (evidence, claim) pairs, from a pretrained checkpoint or one fine-tuned here;
  * entailment, contradiction and neutral mapped to Supported, Contradicted and Insufficient Evidence;
  * argmax by default, with thresholds only from a validation search;
  * mixed-evidence handling, and per-passage softmax scores labelled uncalibrated;
  * Retrieval Failure and Verification Error states;
  * an LLM-judgment fallback that is labelled as not a trained classifier.
* **Controlled Hallucination Test:**
  * the original answer is immutable;
  * the test copy is made by inserting one of 8 reviewed false claims, or by a user-reviewed sentence edit;
  * a transformation record is kept;
  * both answers run through the same pipeline and are shown side by side with highlighting;
  * the actual outcome is reported, including when the claim is not detected.
* **Training scripts:**
  * `prepare_fever` handles split boundaries, leakage checks, balanced subsets and the nearest-page heuristic for NEI evidence.
  * `train_verifier` fine-tunes DeBERTa-v3 or any checkpoint. It reuses an existing NLI head, uses the GPU if available, saves the best checkpoint by validation macro F1, and writes a full training record (or a failed record).
  * `evaluate_verifier` covers experiments A and B, threshold tuning on validation only, and results JSON.
* **RAGTruth evaluation (Experiment C code):**
  * the evidence is each response's own source context;
  * sentence-level and response-level scores, each under strict and lenient rules;
  * QA and Summary tasks by default, with implicit_true spans excluded.
* **Dashboard:**
  * Overview;
  * Ask & verify, with the claim cards and the controlled test;
  * Evaluation: saved results by dataset, a baseline vs fine-tuned comparison, and FEVER end-to-end, RAGTruth and batch-mutation runs;
  * Datasets.

## Tests run and outcomes

* `python -m pytest`: **130 passed**, with torch 2.14 and transformers 5.20 installed. The LLM backends, Wikipedia and the models are mocked, and real network access is blocked.
* `ruff check .`: clean.
* **Full UI flow against mock services:** the app ran through Streamlit's AppTest against mock Ollama and MediaWiki HTTP servers on localhost, using the real `ollama` Python client. After the switch to the API backend it was run again against a mock OpenAI-compatible server (Groq-shaped) for generate → claims → verify. It covered generate → claims → verify, the controlled test in both modes, and RAGTruth evaluation on the real files with the mock verifier. There were no exceptions. The outputs were scripted by the mocks and were discarded; they are not results.
* **Training smoke test** (`tests/test_training_smoke.py`): trains, evaluates and loads a tiny, randomly initialised BERT built locally, with real torch and transformers. It checks that the code paths work, and its numbers mean nothing. It **passed**, including the failed-run record check. It is skipped when torch is not installed.

## Still to run on a machine with internet (and ideally a GPU)

**Done on 2026-10-22/23. See [Experiment results](#experiment-results-run-2026-10-22--23).**

1. Put a Groq key in `.env` (`GROQ_API_KEY`), run `python -m src.cli status`, then `streamlit run app.py`. This exercises the live pipeline. (Or `LLM_PROVIDER=ollama` with `ollama pull llama3.2`.)
2. `python -m src.cli download-nli` gives the baseline verifier.
3. Download `wiki-pages.zip` and `shared_task_dev.jsonl` (or `paper_dev.jsonl`) from fever.ai into `data/fever/`, then run `python -m training.prepare_fever`.
4. **Experiment A:** `evaluate_verifier --model cross-encoder/nli-deberta-v3-xsmall`, with `--tune` on val, then on the held-out split.
5. **Experiment B:** `train_verifier --base-model microsoft/deberta-v3-base --allow-download`, then `evaluate_verifier --model models/fever-verifier`. DeBERTa-v3-base on CPU will be slow; use `--max-train` or a GPU.
6. **Experiment C:** RAGTruth tab or `python -m src.cli eval-ragtruth data/ragtruth/response.jsonl -n 50`.
7. **Experiment D:** the Controlled Hallucination Test in the app, or `python -m src.cli controlled-test`.
