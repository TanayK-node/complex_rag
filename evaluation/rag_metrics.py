"""
RAG-level metrics, as distinct from the pure retrieval metrics in
retrieval_metrics.py (spec section 20 asks for both, evaluated separately).
"""
from __future__ import annotations

import re

from context.context_builder import ContextBundle
from evaluation.retrieval_metrics import is_relevant
from guardrails.groundedness import check_groundedness

_TOKEN_RE = re.compile(r"[a-zA-Z0-9$%]+")
_STOPWORDS = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "or", "for",
              "on", "what", "does", "do", "under", "this", "that", "when", "how"}


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)} - _STOPWORDS


def context_precision(context: ContextBundle, expected_keyword_groups: list[list[str]]) -> float:
    """Of the chunks that made it into the final assembled context, what
    fraction are actually relevant? Low precision = wasted tokens (spec
    section 19's concern), even if the answer still happens to be right."""
    if not context.blocks:
        return 0.0
    relevant = sum(
        1 for b in context.blocks
        if expected_keyword_groups and any(
            all(kw.lower() in b.text.lower() for kw in group) for group in expected_keyword_groups
        )
    )
    return relevant / len(context.blocks)


def context_recall(context: ContextBundle, expected_keyword_groups: list[list[str]]) -> float:
    """Did the assembled context include at least one supporting passage?
    (Same hit-based simplification as retrieval_metrics.hit_at_k.)"""
    if not expected_keyword_groups:
        return 0.0
    return 1.0 if any(
        any(all(kw.lower() in b.text.lower() for kw in group) for group in expected_keyword_groups)
        for b in context.blocks
    ) else 0.0


def answer_relevance(answer: str, question: str) -> float:
    """Crude proxy: how much of the question's own vocabulary does the answer
    engage with. Doesn't verify correctness — a wrong-but-on-topic answer
    still scores well here; that's what faithfulness/correctness are for."""
    q_tokens = _tokens(question)
    a_tokens = _tokens(answer)
    if not q_tokens:
        return 0.0
    return len(q_tokens & a_tokens) / len(q_tokens)


def faithfulness(answer: str, context: ContextBundle) -> float:
    """1.0 if the groundedness checker passes (or the model correctly declared
    insufficient evidence), 0.0 if it flagged unsupported claims/citations."""
    result = check_groundedness(answer, context)
    return 1.0 if result.passed else 0.0


def _norm(text: str) -> str:
    """Lowercase, straighten curly quotes, collapse all whitespace (incl. the
    newlines PDF line-wrapping leaves in the middle of phrases) so a keyword
    like "Modern Slavery Assessment Tool" still matches text that wraps
    mid-phrase."""
    text = (text.replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u201c", '"').replace("\u201d", '"'))
    return " ".join(text.lower().split())


def answer_correctness(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of the expected keywords that appear in the answer (after
    whitespace/quote normalisation). A weak proxy for "is this the right
    answer": a correct answer that paraphrases instead of quoting will score
    low, and a refusal always scores 0 — read the saved answers to tell which."""
    if not expected_keywords:
        return 0.0
    norm_answer = _norm(answer)
    hits = sum(1 for kw in expected_keywords if _norm(kw) in norm_answer)
    return hits / len(expected_keywords)


def correctly_declared_insufficient(answer: str, is_answerable: bool) -> bool:
    declared = answer.strip().upper().startswith("INSUFFICIENT EVIDENCE")
    return declared != is_answerable  # correct iff it declared exactly when it should have