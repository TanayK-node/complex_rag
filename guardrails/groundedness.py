"""
Post-generation verification (spec section 15). This is a lexical-overlap
checker, not a learned faithfulness model (e.g. NLI-based) — it catches the
cases that matter most for a legal document (a citation to a section that
isn't actually in context; a sentence with numbers/figures that appear
nowhere in the retrieved text) without another model dependency. It will
miss subtler unsupported inferences that use only words already present in
context; that's the honest limitation, not a claim of full faithfulness
verification.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from context.context_builder import ContextBundle

_CITATION_RE = re.compile(r"\[Source:\s*([^,\]]+?)(?:,\s*Section\s+([\d.]+))?(?:,\s*Pages?\s+[\d\-]+)?\]")
_NUMBER_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9$%]+")
_STOPWORDS = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "or", "for", "on", "not",
              "does", "do", "under", "this", "that", "shall", "be", "as", "it", "its", "with"}


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)} - _STOPWORDS


@dataclass
class GroundednessResult:
    passed: bool
    is_insufficient_evidence: bool = False
    unsupported_sentences: list[str] = field(default_factory=list)
    bad_citations: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def check_groundedness(answer: str, context: ContextBundle,
                        min_sentence_overlap: float = 0.25) -> GroundednessResult:
    if answer.strip().upper().startswith("INSUFFICIENT EVIDENCE"):
        return GroundednessResult(passed=True, is_insufficient_evidence=True,
                                   notes=["model declared insufficient evidence; nothing to verify"])

    context_tokens = _tokens(context.context_text)
    context_numbers = set(_NUMBER_RE.findall(context.context_text))
    valid_citations = {b.citation for b in context.blocks}
    valid_sections = {b.section_number for b in context.blocks if b.section_number}

    result = GroundednessResult(passed=True)

    # 1) citation validity: every [Source: ...] in the answer must match a citation
    #    that actually came from this context bundle
    for match in _CITATION_RE.finditer(answer):
        full = match.group(0)
        section = match.group(2)
        if full not in valid_citations and (not section or section not in valid_sections):
            result.bad_citations.append(full)

    if not _CITATION_RE.search(answer) and context.blocks:
        result.notes.append("answer makes claims but includes no citation at all")

    # 2) sentence-level support: each substantive sentence should share a
    #    meaningful fraction of its non-stopword tokens with the context
    sentences = [s.strip() for s in re.split(r"(?<=[.;])\s+", answer) if len(s.strip()) > 15]
    for sentence in sentences:
        s_tokens = _tokens(_CITATION_RE.sub("", sentence))
        if not s_tokens:
            continue
        overlap = len(s_tokens & context_tokens) / len(s_tokens)
        if overlap < min_sentence_overlap:
            result.unsupported_sentences.append(sentence)

    # 3) numeric claims are high-stakes in legal text (fees, deadlines, caps) —
    #    flag any number in the answer that doesn't appear anywhere in context
    for num in _NUMBER_RE.findall(answer):
        cleaned = num.replace(",", "")
        if cleaned not in {n.replace(",", "") for n in context_numbers} and len(cleaned) > 1:
            result.unsupported_numbers.append(num)

    result.passed = not (result.bad_citations or result.unsupported_sentences or result.unsupported_numbers)
    return result
