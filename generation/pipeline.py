"""
Orchestrates everything built so far into one `answer_query()` call, and
implements the groundedness retry loop (spec section 15): if verification
fails, retry up to `settings.groundedness_retry_limit` times, first by asking
the same model to fix the specific issues found, then (if issues persist) by
widening the context window before trying again. If retries are exhausted,
return an explicit insufficient-evidence response rather than the last
(still-unverified) attempt.

Also produces a lightweight `trace` (spec section 25's observability ask,
at prototype depth — a full structured logger/latency-per-stage system is
Phase 8's job, not this one's).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from config.settings import settings
from context.context_builder import build_context, ContextBundle
from generation.llm_provider import LLMProvider
from generation.prompt_builder import build_generation_prompt
from guardrails.groundedness import check_groundedness, GroundednessResult
from lexical_search.bm25_index import LexicalIndex
from reranking.base import Reranker
from retrieval.hybrid_retriever import hybrid_retrieve
from schemas.models import Chunk, DocumentTree
from vector_store.faiss_index import VectorStore


@dataclass
class QueryResult:
    question: str
    answer: str
    citations: list[str]
    context: ContextBundle
    groundedness: GroundednessResult
    retries_used: int
    trace: list[dict] = field(default_factory=list)


def answer_query(
    question: str,
    tree: DocumentTree,
    chunks_by_id: dict[str, Chunk],
    vector_store: VectorStore,
    lexical_index: LexicalIndex,
    reranker: Reranker,
    llm: LLMProvider,
) -> QueryResult:
    trace: list[dict] = []

    def _log(stage: str, **kv):
        trace.append({"stage": stage, "t": round(time.time(), 3), **kv})

    _log("query_received", question=question)

    candidates, analysis = hybrid_retrieve(question, tree, chunks_by_id, vector_store, lexical_index)
    _log("retrieval", candidates=len(candidates), explicit_refs=analysis.explicit_section_refs,
         multi_hop=analysis.is_multi_hop, comparison=analysis.is_comparison)

    rerank_k = settings.rerank_top_k
    reranked = reranker.rerank(question, candidates, top_k=rerank_k)
    _log("rerank", kept=len(reranked), top_score=reranked[0].rerank_score if reranked else 0.0)

    context = build_context(reranked, max_tokens=settings.max_context_tokens)
    _log("context_build", tokens=context.total_tokens, blocks=len(context.blocks),
         dropped_for_budget=len(context.dropped_for_budget))

    # Spec section 9: detect insufficient evidence *before* spending a generation
    # call on it, not just after. A reranker score this low means nothing in the
    # candidate pool is a good match for the query — generating from weak context
    # risks exactly the kind of confident-but-wrong answer section 13 forbids.
    # (This threshold is a starting point, not a fixed constant — Phase 6's
    # evaluation framework is where it should actually get tuned against labeled
    # answerable/unanswerable examples rather than hand-picked ones.)
    if not reranked or reranked[0].rerank_score < settings.similarity_threshold:
        _log("insufficient_evidence_low_relevance",
             top_score=reranked[0].rerank_score if reranked else 0.0,
             threshold=settings.similarity_threshold)
        answer = (
            "INSUFFICIENT EVIDENCE: No sufficiently relevant content was found in the "
            "document for this question."
        )
        groundedness = GroundednessResult(passed=True, is_insufficient_evidence=True,
                                           notes=["blocked before generation: below relevance threshold"])
        return QueryResult(question=question, answer=answer, citations=[], context=context,
                            groundedness=groundedness, retries_used=0, trace=trace)

    retries_used = 0
    answer = ""
    groundedness = GroundednessResult(passed=False)

    while True:
        system, user = build_generation_prompt(question, context)

        if retries_used == 1 and not groundedness.passed and not groundedness.is_insufficient_evidence:
            issues = _describe_issues(groundedness)
            user += (f"\n\n[Verification note: a previous draft had these problems — {issues}. "
                     f"Produce a corrected answer that fixes them, still following all the rules above.]")

        response = llm.complete(system, user, max_tokens=600)
        answer = response.text
        _log("generation", attempt=retries_used, input_tokens=response.input_tokens,
             output_tokens=response.output_tokens, model=response.model)

        groundedness = check_groundedness(answer, context)
        _log("groundedness_check", passed=groundedness.passed,
             is_insufficient_evidence=groundedness.is_insufficient_evidence,
             bad_citations=groundedness.bad_citations,
             unsupported_sentences=len(groundedness.unsupported_sentences),
             unsupported_numbers=groundedness.unsupported_numbers)

        if groundedness.passed:
            break
        if retries_used >= settings.groundedness_retry_limit:
            answer = ("INSUFFICIENT EVIDENCE: the system could not produce a fully grounded answer "
                      "within the retry budget. Please review the source document directly for this "
                      "question, or rephrase it.")
            _log("retry_budget_exhausted")
            break

        retries_used += 1
        if retries_used == settings.groundedness_retry_limit:
            # last attempt: widen the context instead of just asking the model to try harder again
            wider_reranked = reranker.rerank(question, candidates, top_k=min(len(candidates), rerank_k * 2))
            context = build_context(wider_reranked, max_tokens=int(settings.max_context_tokens * 1.5))
            _log("context_widened", tokens=context.total_tokens, blocks=len(context.blocks))

    citations = _extract_citations(answer)
    return QueryResult(
        question=question, answer=answer, citations=citations, context=context,
        groundedness=groundedness, retries_used=retries_used, trace=trace,
    )


def _describe_issues(g: GroundednessResult) -> str:
    bits = []
    if g.bad_citations:
        bits.append(f"citations not found in the provided context ({', '.join(g.bad_citations)})")
    if g.unsupported_sentences:
        bits.append(f"{len(g.unsupported_sentences)} sentence(s) with little support in the context")
    if g.unsupported_numbers:
        bits.append(f"number(s) not present anywhere in the context ({', '.join(g.unsupported_numbers)})")
    return "; ".join(bits) or "unspecified groundedness issue"


def _extract_citations(answer: str) -> list[str]:
    import re
    return re.findall(r"\[Source:[^\]]+\]", answer)
