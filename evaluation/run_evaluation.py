"""
Two things this module runs, matching the spec's two asks:

1. run_baseline_comparison() — spec section 21: retrieval quality, latency,
   and token usage for all 5 baselines, on the SAME question set, so the
   comparison actually isolates what each added component contributes.

2. run_rag_evaluation() — spec section 20: full-pipeline (production, i.e.
   baseline 5 + generation) RAG-level metrics — context precision/recall,
   faithfulness, answer relevance/correctness, and whether insufficient-
   evidence questions were correctly recognized as such.

Both operate on the same evaluation.schema.EvalDataset, so one question set
drives everything.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from evaluation.baselines import BASELINE_NAMES, build_all_baselines, retrieve_with_baseline
from evaluation.rag_metrics import (
    answer_correctness, answer_relevance, context_precision, context_recall,
    correctly_declared_insufficient, faithfulness,
)
from evaluation.retrieval_metrics import hit_at_k, is_relevant, ndcg_at_k, precision_at_k, reciprocal_rank
from evaluation.schema import EvalDataset
from generation.llm_provider import LLMProvider
from generation.pipeline import answer_query
from generation.providers import DeterministicMockProvider
from ingestion.pipeline import ingest_pdf
from lexical_search.bm25_index import LexicalIndex
from reranking.heuristic_reranker import get_reranker
from vector_store.embedder import get_embedder
from vector_store.faiss_index import VectorStore


@dataclass
class BaselineResult:
    baseline_id: int
    name: str
    mean_precision_at_5: float
    mean_hit_at_5: float
    mean_mrr: float
    mean_ndcg_at_5: float
    mean_latency_ms: float
    mean_tokens_retrieved: float
    n_questions: int


@dataclass
class RagEvalRow:
    question_id: str
    category: str
    context_precision: float
    context_recall: float
    answer_relevance: float
    faithfulness: float
    answer_correctness: float
    correctly_handled_answerability: bool
    retries_used: int
    used_mock_fallback: bool = False  # True if the configured LLM had already
                                      # failed (e.g. daily quota) and this row's
                                      # generation-dependent metrics come from
                                      # the mock provider, not the real model
    # --- stored for manual comparison (added so you can read, not just score) ---
    question: str = ""
    is_answerable: bool = True
    expected_answer: str = ""          # your hand-written reference (optional, from the JSON)
    expected_keywords: list = field(default_factory=list)
    reference_passage: str = ""        # verbatim text of the document chunk(s) matching expected_keyword_groups
    answer: str = ""                   # what the system actually returned
    answer_source: str = ""            # "answered" | "gate_refusal" | "llm_refusal"
    top_rerank_score: float = 0.0      # best reranker score (compare to similarity_threshold)
    citations: list = field(default_factory=list)
    context_sections: list = field(default_factory=list)   # citations of the blocks sent to the LLM
    context_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    groundedness_passed: bool = True
    groundedness_issues: str = ""
    notes: str = ""


def _reference_passage(chunks, groups, max_chars: int = 700) -> str:
    """Verbatim text of the first chunk matching each keyword group — the
    document's own supporting evidence, so you can judge an answer against
    the source without opening the PDF. Empty for unanswerable questions."""
    parts = []
    for group in groups:
        for c in chunks:
            if is_relevant(c, [group]):
                sec = f"[Section {c.section_number}, p.{c.page_start}] " if c.section_number else f"[p.{c.page_start}] "
                parts.append(sec + " ".join(c.source_text.split())[:max_chars])
                break
    return "\n---\n".join(parts)


def _answer_source(answer: str, trace: list) -> str:
    if any(t["stage"] == "insufficient_evidence_low_relevance" for t in trace):
        return "gate_refusal"      # blocked BEFORE the LLM was called (relevance threshold)
    if answer.strip().upper().startswith("INSUFFICIENT EVIDENCE"):
        return "llm_refusal"       # the LLM (or retry-exhaustion) declined to answer
    return "answered"


@dataclass
class RagEvalResult:
    rows: list[RagEvalRow]
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    overall: dict[str, float] = field(default_factory=dict)


def run_baseline_comparison(pdf_path: str, dataset: EvalDataset, top_k: int = 5) -> list[BaselineResult]:
    # only score questions that actually have a supporting passage to find —
    # "no_answer" questions have no expected_keyword_groups and would trivially
    # score 0 on every retrieval metric regardless of baseline quality
    scored_questions = [q for q in dataset.questions if q.expected_keyword_groups]

    baselines = build_all_baselines(pdf_path, dataset.document_name)
    results = []

    for bid in (1, 2, 3, 4, 5):
        artifacts = baselines[bid]
        precisions, hits, rrs, ndcgs, latencies, tokens = [], [], [], [], [], []

        for q in scored_questions:
            ranked, latency = retrieve_with_baseline(bid, artifacts, q.question, top_k)
            precisions.append(precision_at_k(ranked, q.expected_keyword_groups, top_k))
            hits.append(hit_at_k(ranked, q.expected_keyword_groups, top_k))
            rrs.append(reciprocal_rank(ranked, q.expected_keyword_groups))
            ndcgs.append(ndcg_at_k(ranked, q.expected_keyword_groups, top_k))
            latencies.append(latency * 1000)
            tokens.append(sum(c.token_count for c in ranked[:top_k]))

        n = max(1, len(scored_questions))
        results.append(BaselineResult(
            baseline_id=bid, name=artifacts.name,
            mean_precision_at_5=sum(precisions) / n,
            mean_hit_at_5=sum(hits) / n,
            mean_mrr=sum(rrs) / n,
            mean_ndcg_at_5=sum(ndcgs) / n,
            mean_latency_ms=sum(latencies) / n,
            mean_tokens_retrieved=sum(tokens) / n,
            n_questions=len(scored_questions),
        ))

    return results


def run_rag_evaluation(pdf_path: str, dataset: EvalDataset, llm: LLMProvider) -> RagEvalResult:
    tree, chunks = ingest_pdf(pdf_path, document_name=dataset.document_name)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    vector_store = VectorStore(get_embedder())
    vector_store.build(chunks)
    lexical_index = LexicalIndex()
    lexical_index.build(chunks)
    reranker = get_reranker()

    rows: list[RagEvalRow] = []
    llm_failed = False
    for i, q in enumerate(dataset.questions):
        active_llm = llm
        if llm_failed:
            # once the configured provider has failed (e.g. a daily quota
            # exhausted mid-run), don't keep hammering it for every remaining
            # question — fall back to the mock so the run finishes and you
            # still get retrieval/context metrics (and a clearly-labeled
            # placeholder for the generation-dependent ones) for every
            # question, instead of losing everything after question N.
            active_llm = DeterministicMockProvider()

        t0 = time.time()
        try:
            result = answer_query(q.question, tree, chunks_by_id, vector_store, lexical_index, reranker, active_llm)
        except RuntimeError as e:
            if not llm_failed:
                print(f"\n[WARNING] LLM call failed on question {i + 1}/{len(dataset.questions)} "
                      f"({q.id}): {e}\n  -> falling back to the mock provider for the rest of this "
                      f"run so it still completes. Re-run later (e.g. after a daily quota resets, "
                      f"or with a different model/provider) to get real numbers for the remaining "
                      f"questions.\n")
                llm_failed = True
            active_llm = DeterministicMockProvider()
            result = answer_query(q.question, tree, chunks_by_id, vector_store, lexical_index, reranker, active_llm)
        latency = time.time() - t0

        gen_calls = [t for t in result.trace if t["stage"] == "generation"]
        rerank_stage = next((t for t in result.trace if t["stage"] == "rerank"), {})
        g = result.groundedness
        issues = []
        if g.bad_citations:
            issues.append(f"bad citations: {g.bad_citations}")
        if g.unsupported_numbers:
            issues.append(f"unsupported numbers: {g.unsupported_numbers}")
        if g.unsupported_sentences:
            issues.append(f"{len(g.unsupported_sentences)} weakly-supported sentence(s)")

        rows.append(RagEvalRow(
            question_id=q.id,
            category=q.category,
            context_precision=context_precision(result.context, q.expected_keyword_groups),
            context_recall=context_recall(result.context, q.expected_keyword_groups),
            answer_relevance=answer_relevance(result.answer, q.question),
            faithfulness=faithfulness(result.answer, result.context),
            answer_correctness=answer_correctness(result.answer, q.expected_keywords),
            correctly_handled_answerability=correctly_declared_insufficient(result.answer, q.is_answerable),
            retries_used=result.retries_used,
            used_mock_fallback=llm_failed,
            question=q.question,
            is_answerable=q.is_answerable,
            expected_answer=q.expected_answer,
            expected_keywords=q.expected_keywords,
            reference_passage=_reference_passage(chunks, q.expected_keyword_groups),
            answer=result.answer,
            answer_source=_answer_source(result.answer, result.trace),
            top_rerank_score=float(rerank_stage.get("top_score", 0.0)),
            citations=result.citations,
            context_sections=[b.citation for b in result.context.blocks],
            context_tokens=result.context.total_tokens,
            input_tokens=sum(t.get("input_tokens", 0) for t in gen_calls),
            output_tokens=sum(t.get("output_tokens", 0) for t in gen_calls),
            latency_s=round(latency, 2),
            groundedness_passed=g.passed,
            groundedness_issues="; ".join(issues),
            notes=q.notes,
        ))

    def _agg(rows_subset: list[RagEvalRow]) -> dict[str, float]:
        n = max(1, len(rows_subset))
        answerable = [r for r in rows_subset if r.is_answerable]
        wrong = [r for r in answerable if r.answer_source != "answered"]      # refused although answerable
        answered = [r for r in answerable if r.answer_source == "answered"]
        return {
            "context_precision": sum(r.context_precision for r in rows_subset) / n,
            "context_recall": sum(r.context_recall for r in rows_subset) / n,
            "answer_relevance": sum(r.answer_relevance for r in rows_subset) / n,
            "faithfulness": sum(r.faithfulness for r in rows_subset) / n,
            "answer_correctness": sum(r.answer_correctness for r in rows_subset) / n,
            "answerability_accuracy": sum(1.0 for r in rows_subset if r.correctly_handled_answerability) / n,
            # --- refusal diagnostics (answerable questions only) ---
            "wrongly_refused": float(len(wrong)),
            "wrongly_refused_by_gate": float(sum(1 for r in wrong if r.answer_source == "gate_refusal")),
            "wrongly_refused_by_llm": float(sum(1 for r in wrong if r.answer_source == "llm_refusal")),
            "correctness_when_answered": (sum(r.answer_correctness for r in answered) / len(answered)) if answered else 0.0,
            "n": len(rows_subset),
        }

    by_category: dict[str, dict[str, float]] = {}
    for cat in sorted({r.category for r in rows}):
        by_category[cat] = _agg([r for r in rows if r.category == cat])

    return RagEvalResult(rows=rows, by_category=by_category, overall=_agg(rows))