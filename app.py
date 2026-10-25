"""Streamlit app: Hallucination Detection Using Machine Learning and Evidence Verification.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import json
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import MODELS_DIR, Settings
from src.dataset_eval import evaluate_fever, evaluate_ragtruth
from src.datasets.fever_loader import load_fever_jsonl
from src.datasets.ragtruth_loader import load_ragtruth
from src.evaluation import EvaluationReport
from src.experiments.injection import (
    InjectionError,
    insert_claim,
    load_reviewed_examples,
    run_controlled_test,
    substitute_sentence,
    suggest_replacements,
)
from src.experiments.mutation import (
    MutationValidator,
    ReferenceItem,
    RuleBasedMutator,
    ValidationStatus,
    generate_candidates,
    run_experiment,
)
from src.llm.base import LLMError
from src.llm.factory import PROVIDER_NAMES, PROVIDERS, api_key_for, generation_model, make_llm_client
from src.pipeline import HallucinationPipeline
from src.retrieval.wikipedia import WikipediaRetriever
from src.schemas import Answer, ClaimResult, Outcome, PipelineResult, RetrievalStatus
from src.verification.factory import VerifierChoice, build_verifier
from src.verification.nli_verifier import NLIVerifier, nli_dependencies_available

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
st.set_page_config(page_title="Hallucination Detection", page_icon=":material/fact_check:", layout="wide")

OUTCOME_COLOR = {
    Outcome.SUPPORTED: "green",
    Outcome.CONTRADICTED: "red",
    Outcome.INSUFFICIENT_EVIDENCE: "gray",
    Outcome.RETRIEVAL_FAILURE: "orange",
    Outcome.VERIFICATION_ERROR: "violet",
}
OUTCOME_ICON = {
    Outcome.SUPPORTED: ":material/check_circle:",
    Outcome.CONTRADICTED: ":material/cancel:",
    Outcome.INSUFFICIENT_EVIDENCE: ":material/help:",
    Outcome.RETRIEVAL_FAILURE: ":material/cloud_off:",
    Outcome.VERIFICATION_ERROR: ":material/error:",
}


def badge(outcome: Outcome) -> str:
    return f":{OUTCOME_COLOR[outcome]}-badge[{OUTCOME_ICON[outcome]} {outcome.value}]"


# ------------------------------------------------------------------- styling
GLASS_CSS = Path(__file__).parent / "assets" / "glass.css"
GLASS_TOKENS = {
    "light": {
        "--bg": "#f4f5fb", "--blob-1": "rgba(99,102,241,0.20)", "--blob-2": "rgba(20,184,166,0.16)",
        "--blob-3": "rgba(236,72,153,0.12)", "--glass-bg": "rgba(255,255,255,0.55)",
        "--glass-bg-strong": "rgba(255,255,255,0.70)", "--glass-border": "rgba(255,255,255,0.75)",
        "--glass-hover": "rgba(255,255,255,0.45)", "--glass-shadow": "0 8px 32px rgba(31,38,92,0.08)",
        "--field-bg": "rgba(255,255,255,0.60)", "--tab-active": "rgba(255,255,255,0.95)",
        "--text-strong": "#1c1d2b", "--accent": "#5b5bd6", "--accent-2": "#8b5cf6",
        "--accent-glow": "rgba(91,91,214,0.30)", "--mark": "rgba(239,68,68,0.22)",
        "--line": "rgba(28,29,43,0.12)",
    },
    "dark": {
        "--bg": "#0b0d17", "--blob-1": "rgba(99,102,241,0.30)", "--blob-2": "rgba(20,184,166,0.16)",
        "--blob-3": "rgba(168,85,247,0.18)", "--glass-bg": "rgba(255,255,255,0.05)",
        "--glass-bg-strong": "rgba(14,17,30,0.60)", "--glass-border": "rgba(255,255,255,0.10)",
        "--glass-hover": "rgba(255,255,255,0.04)", "--glass-shadow": "0 8px 32px rgba(0,0,0,0.35)",
        "--field-bg": "rgba(255,255,255,0.06)", "--tab-active": "rgba(255,255,255,0.12)",
        "--text-strong": "#f4f5ff", "--accent": "#6d6de8", "--accent-2": "#a78bfa",
        "--accent-glow": "rgba(124,124,240,0.35)", "--mark": "rgba(248,113,113,0.30)",
        "--line": "rgba(255,255,255,0.14)",
    },
}


def inject_style() -> None:
    """Frosted-glass surfaces over a soft gradient; tokens follow the active light/dark theme."""
    mode = "dark" if st.context.theme.type == "dark" else "light"
    tokens = "".join(f"{k}:{v};" for k, v in GLASS_TOKENS[mode].items())
    st.html(f"<style>:root{{{tokens}}}\n{GLASS_CSS.read_text(encoding='utf-8')}</style>")


# ------------------------------------------------------------------ settings
def threshold_files(settings: Settings) -> list[Path]:
    return sorted(settings.results_dir.glob("thresholds_*.json")) if settings.results_dir.is_dir() else []


def sidebar_settings() -> Settings:
    base = Settings.from_env()
    st.sidebar.header("Settings")
    provider = st.sidebar.selectbox(
        "Generation backend", PROVIDER_NAMES,
        index=PROVIDER_NAMES.index(base.llm_provider) if base.llm_provider in PROVIDER_NAMES else 0,
        help="ollama: local Llama via Ollama. groq / openrouter / openai / custom: hosted "
        "OpenAI-compatible API; the key is read from .env and never shown here.",
    )
    host, model, base_url = base.ollama_host, base.ollama_model, base.llm_base_url
    llm_model = base.llm_model
    if provider == "ollama":
        host = st.sidebar.text_input("Ollama host", base.ollama_host)
        models = cached_llm_status("ollama", host, base.ollama_model, "")["models"]
        if models:
            default = next((i for i, m in enumerate(models)
                            if m.split(":")[0] == base.ollama_model.split(":")[0]), 0)
            model = st.sidebar.selectbox("Generation model (installed)", models, index=default)
        else:
            model = st.sidebar.text_input("Generation model", base.ollama_model)
    else:
        preset = PROVIDERS.get(provider)
        llm_model = st.sidebar.text_input(
            "Generation model", base.llm_model or (preset.default_model if preset else ""))
        if provider == "custom":
            base_url = st.sidebar.text_input("API base URL", base.llm_base_url)
        key_env = preset.key_env if preset else "LLM_API_KEY"
        if api_key_for(provider):
            st.sidebar.caption(f"API key found in {key_env} (not displayed).")
        else:
            st.sidebar.warning(f"No API key: set {key_env} in .env and restart.")
        if preset:
            st.sidebar.caption(preset.note)
    extractor = st.sidebar.selectbox(
        "Claim extraction", ["spacy", "llm"], index=0 if base.extractor != "llm" else 1,
        help="spacy: one candidate claim per sentence (Sentencizer), multi-assertion sentences flagged. "
        "llm: spaCy sentences decomposed into sub-claims by the generation LLM (falls back to spacy with a warning).",
    )
    verifier = st.sidebar.selectbox(
        "Verifier", ["auto", "nli", "llm"], index=["auto", "nli", "llm"].index(base.verifier),
        help="nli: NLI classifier. llm: generation-LLM judgment fallback (not a trained classifier). "
        "auto: NLI if it loads, otherwise LLM (shown on the Overview tab).",
    )
    local_ckpts = [str(p) for p in sorted(MODELS_DIR.glob("*")) if (p / "config.json").is_file()]
    nli_options = list(dict.fromkeys([base.nli_model, *local_ckpts]))
    nli_model = st.sidebar.selectbox("NLI model (Hugging Face id or local checkpoint)", nli_options,
                                     help="Fine-tuned checkpoints from training/train_verifier.py appear here.")
    sup = con = None
    files = threshold_files(base)
    if files and st.sidebar.checkbox("Apply thresholds tuned on validation data"):
        f = st.sidebar.selectbox("Thresholds file", files, format_func=lambda p: p.name)
        th = json.loads(f.read_text())
        sup, con = th.get("support_threshold"), th.get("contradiction_threshold")
        st.sidebar.caption(f"support={sup}, contradiction={con} (tuned on '{th.get('tuned_on_split')}', "
                           f"n={th.get('n')}, model {th.get('model')})")
    else:
        st.sidebar.caption("NLI decision: argmax per passage (no thresholds).")
    max_claims = st.sidebar.slider("Max claims per answer", 1, 25, base.max_claims)
    if st.sidebar.button("Re-check services"):
        st.cache_data.clear()
    return replace(base, llm_provider=provider, llm_model=llm_model, llm_base_url=base_url.rstrip("/"),
                   ollama_host=host.rstrip("/"), ollama_model=model, extractor=extractor,
                   verifier=verifier, nli_model=nli_model, nli_support_threshold=sup,
                   nli_contradiction_threshold=con, max_claims=max_claims)


@st.cache_data(ttl=30, show_spinner=False)
def cached_llm_status(provider: str, host: str, model: str, base_url: str) -> dict:
    settings = replace(Settings.from_env(), llm_provider=provider, ollama_host=host, ollama_model=model,
                       llm_model=model if provider != "ollama" else "", llm_base_url=base_url,
                       ollama_timeout_s=5, llm_timeout_s=10)
    try:
        s = make_llm_client(settings).status()
    except ValueError as exc:
        return {"reachable": False, "models": [], "model_available": False, "error": str(exc)}
    return {"reachable": s.reachable, "models": s.models, "model_available": s.model_available, "error": s.error}


@st.cache_data(ttl=60, show_spinner=False)
def cached_wiki_status(api_url: str, user_agent: str) -> tuple[bool, str]:
    return WikipediaRetriever(api_url, user_agent, timeout_s=5, max_retries=1).check_connectivity()


@st.cache_resource(show_spinner="Loading verifier...")
def cached_verifier(mode: str, nli_model: str, provider: str, host: str, model: str, base_url: str,
                    seed: int) -> VerifierChoice:
    settings = replace(Settings.from_env(), verifier=mode, nli_model=nli_model, llm_provider=provider,
                       ollama_host=host, ollama_model=model, llm_model=model if provider != "ollama" else "",
                       llm_base_url=base_url, random_seed=seed)
    try:
        return build_verifier(settings, make_llm_client(settings))
    except Exception as exc:  # noqa: BLE001 - shown in the UI
        return VerifierChoice(None, "unavailable", [str(exc)])  # type: ignore[arg-type]


@st.cache_resource(show_spinner=False)
def cached_retriever(api_url: str, user_agent: str) -> WikipediaRetriever:
    s = Settings.from_env()
    return WikipediaRetriever(api_url, user_agent, s.wiki_timeout_s, s.wiki_max_retries, s.wiki_search_results,
                              s.wiki_passages_per_claim, s.wiki_min_interval_s, s.wiki_cache_dir)


def get_pipeline(settings: Settings) -> tuple[HallucinationPipeline | None, VerifierChoice]:
    choice = cached_verifier(settings.verifier, settings.nli_model, settings.llm_provider, settings.ollama_host,
                             generation_model(settings), settings.llm_base_url, settings.random_seed)
    if choice.verifier is None:
        return None, choice
    if isinstance(choice.verifier, NLIVerifier):
        choice.verifier.support_threshold = settings.nli_support_threshold
        choice.verifier.contradiction_threshold = settings.nli_contradiction_threshold
    try:
        client = make_llm_client(settings)
    except ValueError as exc:
        return None, VerifierChoice(None, "unavailable", [str(exc)])  # type: ignore[arg-type]
    retriever = cached_retriever(settings.wiki_api_url, settings.wiki_user_agent)
    return HallucinationPipeline(client, retriever, choice.verifier, settings.extractor, settings.max_claims), choice


def make_endpoint(settings: Settings) -> str:
    preset = PROVIDERS.get(settings.llm_provider)
    return settings.llm_base_url or (preset.base_url if preset else "(not set)")


# ------------------------------------------------------------------ overview
def render_overview(settings: Settings, choice: VerifierChoice) -> None:
    st.markdown(
        "A pretrained LLM (Groq API by default, or local Llama via Ollama) answers a question. The answer "
        "is split into candidate claims (spaCy), "
        "evidence for each claim is retrieved from Wikipedia, and a separate verifier labels each claim "
        "**Supported**, **Contradicted** or **Insufficient Evidence** from the evidence alone. This is "
        "evidence-based checking: it can flag likely hallucinations, but it cannot guarantee that "
        "any statement is correct."
    )
    o = cached_llm_status(settings.llm_provider, settings.ollama_host, generation_model(settings),
                          settings.llm_base_url)
    wiki_ok, wiki_msg = cached_wiki_status(settings.wiki_api_url, settings.wiki_user_agent)
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True, key="glass_llm"):
        st.markdown(f"**:material/smart_toy: Answer generation ({settings.llm_provider})**")
        if o["reachable"] and o["model_available"]:
            st.success(f"Connected. Model `{generation_model(settings)}` is available.")
        elif o["reachable"]:
            st.warning(o["error"])
        else:
            st.error(o["error"])
        endpoint = settings.ollama_host if settings.llm_provider == "ollama" else make_endpoint(settings)
        st.caption(f"Endpoint: {endpoint}")
    with c2.container(border=True, key="glass_wiki"):
        st.markdown("**:material/travel_explore: Evidence retrieval (Wikipedia API)**")
        (st.success if wiki_ok else st.error)(wiki_msg)
        st.caption("Without network access every claim shows Retrieval Failure; nothing is assumed true or false.")
    with c3.container(border=True, key="glass_verifier"):
        st.markdown("**:material/verified: Verifier**")
        if choice.verifier is None:
            st.error(f"Unavailable: {choice.notes[0] if choice.notes else 'unknown error'}")
        elif choice.kind == "llm" and not o["model_available"]:
            st.warning(f"{choice.verifier.method}. It needs the generation backend, which is not usable right now.")
        else:
            st.success(choice.verifier.method)
        if choice.verifier is not None:
            for note in choice.notes:
                st.warning(note)
        st.caption(f"NLI packages installed: {'yes' if nli_dependencies_available() else 'no'}")
    st.info(
        "**Research integrity.** Llama is used for inference only. A pretrained NLI model is not trained by "
        "this project; a checkpoint counts as fine-tuned only if its training record says the run completed. "
        "Model scores are uncalibrated softmax outputs. Metrics appear only when computed from real labels, "
        "and FEVER, RAGTruth and controlled tests are reported separately.",
        icon=":material/shield:",
    )


# ------------------------------------------------------------------ results
def highlight(text: str, start: int | None = None, end: int | None = None) -> str:
    """HTML-escaped text with an optional highlighted span."""
    if start is None or end is None:
        return html.escape(text).replace("\n", "<br>")
    return (html.escape(text[:start]) + "<mark class='glass-mark'>"
            + html.escape(text[start:end]) + "</mark>" + html.escape(text[end:])).replace("\n", "<br>")


def render_claim(cr: ClaimResult, highlight_claim: bool = False) -> None:
    v, r, c = cr.verdict, cr.retrieval, cr.claim
    title = f"{c.id} · {cr.outcome.value} · {c.text}"
    with st.expander(("INJECTED · " if highlight_claim else "") + title, expanded=True,
                     icon=":material/warning:" if highlight_claim else OUTCOME_ICON[cr.outcome]):
        st.markdown(f"{badge(cr.outcome)} &nbsp; `{c.id}` &nbsp; {c.text}")
        if c.multi_assertion:
            st.caption("Flagged: may contain several assertions (" + ", ".join(c.flags) + "). Verified as a whole.")
        if v.mixed_evidence:
            st.warning("Mixed evidence: some passages support and others contradict this claim; no verdict forced.")
        if cr.outcome is Outcome.RETRIEVAL_FAILURE:
            st.error(f"Retrieval failure: {r.error}")
        if v.error:
            st.error(f"Verification error: {v.error}")
        st.markdown("**Retrieved evidence** (verbatim source text):")
        if v.evidence:
            for p in v.evidence:
                st.markdown(f"[{p.title}]({p.url}) · passage {p.passage_index}" if p.url else p.title)
                st.text(p.text[:1500])
        elif r.status is RetrievalStatus.OK:
            st.caption("Retrieved: " + ", ".join(f"[{p.title}]({p.url})" for p in r.passages))
        else:
            st.caption("No evidence passages.")
        st.markdown("**Verifier explanation** (generated by this app, not source text):")
        st.write(v.explanation)
        st.caption(f"Method: {v.method} · Retrieval status: {r.status.value}" + (" (cached)" if r.from_cache else ""))
        per = v.raw_scores.get("per_passage") if v.raw_scores else None
        if per:
            st.caption(v.score_note)
            st.dataframe(pd.DataFrame(per), hide_index=True, width="stretch")
        elif v.raw_scores:
            st.caption(v.score_note)
            st.json(v.raw_scores)


def render_summary(result: PipelineResult) -> None:
    s = result.summary
    counts = {o: sum(c.outcome is o for c in result.claims) for o in Outcome}
    cols = st.columns(6)
    cols[0].metric("Claims checked", s.total_claims)
    for col, o in zip(cols[1:], Outcome, strict=True):
        col.metric(o.value, counts[o])
    share = s.contradicted_share_of_evidenced
    text = f"Extraction: **{result.extraction_method}**. Verifier: **{result.verifier_method}**. "
    if share is not None:
        text += (f"{s.contradicted} of {s.claims_with_evidence} claims with retrieved evidence were contradicted "
                 f"({share:.0%}). This is a descriptive share, not a hallucination probability; claims without "
                 "evidence are excluded from the denominator.")
    else:
        text += "No claim had usable evidence, so no contradiction share is reported."
    st.markdown(text)


def render_result(result: PipelineResult, title: str, injected: tuple[int, int] | None = None) -> None:
    a = result.answer
    st.subheader(title)
    if a.source == "llm":
        st.caption(f"Generated by `{a.model}` via {a.provider or 'the LLM backend'}. Stored unmodified.")
    elif a.source == "controlled-test":
        st.caption("CONTROLLED TEST COPY: contains a deliberately injected false claim. Not model output.")
    else:
        st.caption("Pasted by the user (not generated in this session).")
    with st.container(border=True, key="glass_answer"):
        st.markdown(highlight(a.text, *(injected or (None, None))), unsafe_allow_html=True)
    for w in result.warnings:
        st.warning(w)
    skipped = [s for s in result.sentences if s.skip_reason]
    if skipped:
        st.caption("Not checked: " + "; ".join(f"{s.id} ({s.skip_reason})" for s in skipped))
    render_summary(result)
    for cr in result.claims:
        hit = bool(injected and cr.claim.span_start is not None
                   and cr.claim.span_start < injected[1] and injected[0] < cr.claim.span_end)
        render_claim(cr, hit)


def render_ask(settings: Settings, pipeline: HallucinationPipeline | None) -> None:
    question = st.text_area("Question", placeholder="e.g. When was the Eiffel Tower built and how tall is it?",
                            height=80)
    with st.expander("Or verify an existing answer (paste text instead of generating)"):
        pasted = st.text_area("Answer text to verify", height=120)
    c1, c2 = st.columns(2)
    gen = c1.button("Generate answer", type="primary", icon=":material/auto_awesome:",
                    disabled=pipeline is None or not question.strip())
    ver = c2.button("Verify pasted answer", icon=":material/fact_check:",
                    disabled=pipeline is None or not pasted.strip())
    if pipeline is None:
        st.error("No verifier is available; see the Overview tab.")
        return
    if gen or ver:
        st.session_state.pop("controlled", None)
        with st.status("Running pipeline...", expanded=True) as status:
            try:
                if gen:
                    st.write(f"Generating an answer with `{generation_model(settings)}` ({settings.llm_provider})...")
                    answer = pipeline.generate_answer(question)
                else:
                    answer = Answer(question or "(none)", pasted.strip(), "n/a", source="user")
                st.write("Extracting claims, retrieving evidence and verifying...")
                bar = st.progress(0.0)

                def progress(i: int, n: int, text: str) -> None:
                    bar.progress((i + 1) / max(n, 1), text=f"Claim {i + 1}/{n}: {text[:80]}")

                st.session_state["result"] = pipeline.analyze(answer, progress)
                status.update(label="Done", state="complete")
            except LLMError as exc:
                status.update(label="Answer generation failed", state="error")
                st.error(f"{type(exc).__name__}: {exc}")
                st.session_state.pop("result", None)

    result: PipelineResult | None = st.session_state.get("result")
    if result is None:
        return
    render_result(result, "Original answer" + (" (LLM output)" if result.answer.source == "llm" else ""))
    st.download_button("Download results (JSON)", json.dumps(result.to_dict(), indent=2),
                       "verification_result.json", "application/json", icon=":material/download:")
    st.space("large")
    render_controlled_test(settings, pipeline, result)


# ------------------------------------------------------ controlled test (UI)
def render_controlled_test(settings: Settings, pipeline: HallucinationPipeline, result: PipelineResult) -> None:
    st.subheader("Controlled Hallucination Test (optional)")
    st.caption("Builds a separate test copy of the answer with one deliberately false claim and runs both "
               "through the same pipeline. The original answer above is never changed.")
    answer = result.answer
    mode = st.radio("How to create the false claim",
                    ["Insert a reviewed false claim", "Edit one sentence of the answer"], horizontal=True)
    injection = None
    try:
        if mode.startswith("Insert"):
            examples = load_reviewed_examples()
            ex = st.selectbox("Reviewed example", examples, format_func=lambda e: f"{e.topic}: {e.false_claim}")
            st.caption(f"Change: {ex.transformation}. True version: \"{ex.true_claim}\". "
                       f"Check against [{ex.topic} on Wikipedia]({ex.reference_url}).")
            if st.button("Run Controlled Hallucination Test", key="ct_insert"):
                injection = insert_claim(answer, ex.false_claim, ex.transformation, f"reviewed example '{ex.id}'")
        else:
            candidates = [s for s in result.sentences if s.skip_reason is None]
            if not candidates:
                st.info("The answer has no factual sentences to edit.")
                return
            sent = st.selectbox("Sentence to replace", candidates, format_func=lambda s: f"{s.id}: {s.text}")
            suggestions = suggest_replacements(sent.text, settings.random_seed)
            options = ["(write my own)"] + [f"{t}  [{d}]" for t, d in suggestions]
            pick = st.selectbox("Start from a rule-based suggestion (review it!)", options)
            start_text = sent.text if pick == options[0] else suggestions[options.index(pick) - 1][0]
            new_text = st.text_area("False version of the sentence (you review and edit this)", start_text,
                                    key=f"ct_edit_{sent.id}_{pick}")
            desc = st.text_input("Describe the change", suggestions[options.index(pick) - 1][1]
                                 if pick != options[0] else "")
            if st.button("Run Controlled Hallucination Test", key="ct_edit"):
                injection = substitute_sentence(answer, sent, new_text, desc or "user edit",
                                                "user-reviewed sentence edit")
    except InjectionError as exc:
        st.error(str(exc))
    if injection is not None:
        with st.spinner("Verifying the test copy with the same pipeline..."):
            st.session_state["controlled"] = run_controlled_test(pipeline, answer, injection, result)
    ct = st.session_state.get("controlled")
    if ct is None or ct.injection.original_answer != answer.text:
        return
    inj = ct.injection
    st.markdown(f"**Transformation:** {inj.transformation} · origin: {inj.origin} · mode: {inj.mode}")
    if ct.injected_claims:
        outcomes = ", ".join(c.outcome.value for c in ct.injected_claims)
        if ct.detected:
            st.success(f"The injected claim was labelled: {outcomes}. Detected as contradicted.")
        else:
            st.warning(f"The injected claim was labelled: {outcomes}. It was NOT detected as contradicted.")
    else:
        st.warning("The injected text was not extracted as a checkable claim, so it could not be verified.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Original answer** (unmodified)")
        with st.container(border=True, key="glass_ct_original"):
            st.markdown(highlight(inj.original_answer), unsafe_allow_html=True)
    with right:
        st.markdown("**Test copy** (contains the injected claim, highlighted)")
        with st.container(border=True, key="glass_ct_modified"):
            st.markdown(highlight(inj.modified_answer, inj.injected_start, inj.injected_end), unsafe_allow_html=True)
    injected_ids = {c.claim.id for c in ct.injected_claims}
    rows = []
    for side, res in (("original", ct.original), ("test copy", ct.modified)):
        for c in res.claims:
            hit = side == "test copy" and c.claim.id in injected_ids
            rows.append({"answer": side, "id": c.claim.id, "claim": c.claim.text, "outcome": c.outcome.value,
                         "injected": "yes" if hit else ""})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("Claim details for the test copy"):
        for cr in ct.modified.claims:
            render_claim(cr, cr.claim.id in injected_ids)


# ---------------------------------------------------------------- evaluation
def render_report(report: EvaluationReport | dict, key: str = "") -> None:
    r = report.to_dict() if isinstance(report, EvaluationReport) else report
    cfg = r.get("config", {})
    st.markdown(f"**{r['name']}** · dataset **{cfg.get('dataset', '?')}** · split `{cfg.get('split', '?')}` · "
                f"n = {r['n']} · {r.get('created_at', '')}")
    for w in r.get("warnings", []):
        st.warning(w)
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{r['accuracy']:.3f}")
    cols[1].metric("Macro precision", f"{r['macro_precision']:.3f}")
    cols[2].metric("Macro recall", f"{r['macro_recall']:.3f}")
    cols[3].metric("Macro F1", f"{r['macro_f1']:.3f}")
    c1, c2 = st.columns(2)
    c1.markdown("Confusion matrix (rows: gold, columns: predicted)")
    c1.dataframe(pd.DataFrame(r["confusion_matrix"], index=r["labels"], columns=r["labels"]))
    c2.markdown("Per-class metrics")
    c2.dataframe(pd.DataFrame(r["per_class"]).T)
    st.caption(f"Gold distribution: {r['true_distribution']} · Predicted: {r['pred_distribution']}")
    with st.expander("Experiment configuration"):
        st.json(cfg)


def saved_results(settings: Settings) -> list[tuple[Path, dict]]:
    out = []
    if settings.results_dir.is_dir():
        for f in sorted(settings.results_dir.glob("*.json"), reverse=True):
            try:
                d = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(d, dict) and "confusion_matrix" in d:
                out.append((f, d))
    return out


def render_saved_results(settings: Settings) -> None:
    results = saved_results(settings)
    if not results:
        st.info("No saved evaluation results yet. Run an evaluation below or a training/ script; results are "
                "written to `results/`. Nothing is shown until a real run has produced it.")
        return
    table = pd.DataFrame([{
        "file": f.name, "dataset": d["config"].get("dataset", "?"),
        "experiment": d["config"].get("experiment", d["name"]),
        "model": str(d["config"].get("model", ""))[:60], "split": d["config"].get("split", ""),
        "n": d["n"], "accuracy": round(d["accuracy"], 4), "macro F1": round(d["macro_f1"], 4),
        "created": d.get("created_at", ""),
    } for f, d in results])
    for ds in table["dataset"].unique():
        st.markdown(f"**{ds}**")
        st.dataframe(table[table["dataset"] == ds].drop(columns=["dataset"]), hide_index=True, width="stretch")
    fever = table[(table["dataset"] == "FEVER") & table["experiment"].str.startswith(("A:", "B:"))]
    for split in fever["split"].unique():
        sub = fever[fever["split"] == split]
        if sub["experiment"].str.startswith("A:").any() and sub["experiment"].str.startswith("B:").any():
            st.markdown(f"**Baseline vs fine-tuned on FEVER `{split}`** (latest of each)")
            st.dataframe(sub.groupby("experiment").head(1)[["experiment", "model", "n", "accuracy", "macro F1"]],
                         hide_index=True)
    pick = st.selectbox("Show details", [f for f, _ in results], format_func=lambda p: p.name)
    render_report(next(d for f, d in results if f == pick))


def list_files(directory: Path, pattern: str) -> list[Path]:
    return sorted(directory.glob(pattern)) if directory.is_dir() else []


def render_evaluation(settings: Settings, pipeline: HallucinationPipeline | None) -> None:
    st.caption("Every number here comes from a real run on real labels. FEVER, RAGTruth and controlled tests "
               "are separate tasks and are never combined into one score.")
    t_saved, t_fever, t_rag, t_mut = st.tabs(
        ["Saved results", "FEVER (end-to-end)", "RAGTruth", "Controlled mutations"])
    with t_saved:
        render_saved_results(settings)
        st.markdown("FEVER verifier experiments A (pretrained baseline) and B (fine-tuned) run from the command "
                    "line with `training/` scripts; see the README.")
    if pipeline is None:
        for t in (t_fever, t_rag, t_mut):
            t.error("No verifier is available; see the Overview tab.")
        return

    with t_fever:
        st.caption("Claims are verified with live Wikipedia retrieval (today's Wikipedia, not FEVER's 2017 dump).")
        files = [f for f in list_files(settings.data_dir / "fever", "*.jsonl")]
        if not files:
            st.info("No FEVER files found in `data/fever/`. See the Datasets tab.")
        else:
            path = st.selectbox("FEVER file", files, format_func=lambda p: p.name)
            n = st.number_input("Sample size", 1, 2000, 20, key="fever_n")
            if path.name == "train.jsonl":
                st.warning("This is the training split. Use it only to sanity-check the pipeline; it is not a "
                           "held-out evaluation for a model fine-tuned on it.")
            if st.button("Run FEVER end-to-end evaluation"):
                examples, rep = load_fever_jsonl(path)
                bar = st.progress(0.0)
                run = evaluate_fever(examples, pipeline, int(n), settings.random_seed, rep.split,
                                     lambda i, t: bar.progress((i + 1) / t))
                st.session_state["fever_run"] = run
                if run.report:
                    run.report.save(settings.results_dir)
            run = st.session_state.get("fever_run")
            if run:
                if run.report:
                    render_report(run.report)
                else:
                    for w in run.warnings:
                        st.warning(w)
                st.dataframe(pd.DataFrame(run.rows), width="stretch", hide_index=True)

    with t_rag:
        resp = [f for f in list_files(settings.data_dir / "ragtruth", "*.jsonl") if "source" not in f.name]
        if not resp:
            st.info("No RAGTruth files found in `data/ragtruth/`. See the Datasets tab.")
        else:
            path = st.selectbox("Response file", resp, format_func=lambda p: p.name)
            c1, c2, c3 = st.columns(3)
            split = c1.selectbox("Split", ["test", "train", "(all)"])
            tasks = c2.multiselect("Tasks", ["QA", "Summary", "Data2txt"], ["QA", "Summary"])
            n = c3.number_input("Responses", 1, 3000, 10, key="rag_n")
            implicit = st.checkbox("Count spans marked implicit_true as hallucinations", False)
            st.caption("Evidence is each response's own source context. Sentence labels come from overlapping "
                       "annotated spans (see README).")
            if st.button("Run RAGTruth evaluation"):
                examples, _ = load_ragtruth(path)
                bar = st.progress(0.0)
                run = evaluate_ragtruth(examples, pipeline, int(n), settings.random_seed,
                                        None if split == "(all)" else split, tuple(tasks) or None, implicit,
                                        lambda i, t: bar.progress((i + 1) / t))
                st.session_state["rag_run"] = run
                if run.report:
                    run.report.save(settings.results_dir)
                    for r in run.extra_reports.values():
                        r.save(settings.results_dir)
            run = st.session_state.get("rag_run")
            if run:
                if run.report:
                    render_report(run.report)
                    for r in run.extra_reports.values():
                        render_report(r)
                else:
                    for w in run.warnings:
                        st.warning(w)
                st.dataframe(pd.DataFrame(run.rows), width="stretch", hide_index=True)

    with t_mut:
        st.markdown("Batch version of the controlled test: claims with reference evidence are mutated "
                    "(date/number/negation), each mutation is validated against the evidence, and the "
                    "detector sees only claim text. Upload JSONL with `claim` and `evidence` fields.")
        up = st.file_uploader("Claim/evidence JSONL", type=["jsonl", "json"])
        items: list[ReferenceItem] = []
        if up is not None:
            for line in up.getvalue().decode("utf-8").splitlines():
                try:
                    row = json.loads(line)
                    items.append(ReferenceItem(row["claim"], row["evidence"], row.get("title", ""), row.get("url", "")))
                except (ValueError, KeyError, TypeError):
                    continue
        rule_only = st.radio("Validation", ["deterministic checks + verifier", "deterministic checks only"]) \
            == "deterministic checks only"
        if items and st.button("Run controlled mutations"):
            with st.spinner("Generating, validating and detecting..."):
                cands, warns = generate_candidates(items, RuleBasedMutator(settings.random_seed), None,
                                                   MutationValidator(None if rule_only else pipeline.verifier))
                res = run_experiment(items, cands, pipeline.verify_claim_text,
                                     {"dataset": "Controlled mutations", "experiment": "D: batch mutations",
                                      "model": pipeline.verifier.method, "split": "user-supplied",
                                      "validator_same_as_detector": not rule_only}, settings.random_seed)
            res.warnings = warns + res.warnings
            if res.report:
                res.report.save(settings.results_dir)
            st.dataframe(pd.DataFrame([{"original": c.original_claim, "mutated": c.mutated_claim,
                                        "type": c.mutation_type.value, "validation": c.validation_status.value,
                                        "reason": c.validation_reason} for c in res.candidates]),
                         width="stretch", hide_index=True)
            n_valid = sum(c.validation_status is ValidationStatus.VALIDATED for c in res.candidates)
            st.caption(f"{n_valid} of {len(res.candidates)} mutations validated; only those were tested.")
            if res.detection_rate is not None:
                st.metric("Detection rate on validated mutations", f"{res.detection_rate:.0%}")
            if res.report:
                render_report(res.report)


# ------------------------------------------------------------------ datasets
def render_datasets(settings: Settings) -> None:
    fdir, rdir = settings.data_dir / "fever", settings.data_dir / "ragtruth"
    c1, c2 = st.columns(2)
    with c1.container(border=True, key="glass_fever"):
        st.markdown("**:material/dataset: FEVER** (`data/fever/`)")
        for f in list_files(fdir, "*.jsonl"):
            st.write(f"- {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
        wiki = fdir / "wiki-pages"
        st.write(f"- wiki-pages dump: {'present' if wiki.is_dir() else 'missing (needed to build evidence text)'}")
        prep = fdir / "prepared" / "prepare_report.json"
        if prep.is_file():
            with st.expander("prepare_fever report"):
                st.json(json.loads(prep.read_text()))
        st.caption("Official source: https://fever.ai/dataset/fever.html (train, shared_task_dev, wiki-pages).")
    with c2.container(border=True, key="glass_ragtruth"):
        st.markdown("**:material/dataset: RAGTruth** (`data/ragtruth/`)")
        for f in list_files(rdir, "*.jsonl"):
            st.write(f"- {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
        st.caption("Source: https://github.com/ParticleMedia/RAGTruth (MIT licence), files from `dataset/`.")
    report = settings.results_dir / "dataset_report.json"
    if report.is_file():
        st.markdown("**Dataset report** (`python -m src.cli dataset-report`, counts from the actual files)")
        st.json(json.loads(report.read_text()), expanded=False)
    st.caption("Datasets are never downloaded automatically. The app works without them.")


# ---------------------------------------------------------------------- main
def main() -> None:
    inject_style()
    with st.container(border=True, key="glass_hero"):
        st.title("Hallucination Detection Using Machine Learning and Evidence Verification", anchor=False)
        st.markdown(":gray-badge[Research prototype] &nbsp; :blue-badge[:material/smart_toy: LLM answer] → "
                    ":blue-badge[:material/segment: spaCy claims] → :blue-badge[:material/travel_explore: "
                    "Wikipedia evidence] → :blue-badge[:material/verified: NLI verification]")
    settings = sidebar_settings()
    pipeline, choice = get_pipeline(settings)
    t1, t2, t3, t4 = st.tabs([":material/dashboard: Overview", ":material/forum: Ask & verify",
                              ":material/query_stats: Evaluation", ":material/database: Datasets"])
    with t1:
        render_overview(settings, choice)
    with t2:
        render_ask(settings, pipeline)
    with t3:
        render_evaluation(settings, pipeline)
    with t4:
        render_datasets(settings)


main()
