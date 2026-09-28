"""
Stage before retrieval (spec section 7). Cheap, rule-based NLP — not because a
learned intent classifier wouldn't do better, but because for a first pass over
legal queries, explicit section-number extraction and a small set of intent
keyword buckets already resolve the cases the spec calls out (termination,
payment, comparison, multi-hop). This is the natural seam to add an
LLM-based query rewriter/decomposer later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ingestion.structure_detector import NUMBER_TOKEN_RE

_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "in", "and", "or", "for", "on",
    "what", "when", "how", "does", "do", "can", "under", "which", "if", "be",
    "this", "that", "with", "at", "by", "as", "it", "its",
}

# Intent buckets: query keyword -> candidate section topics to prioritize.
# This is what lets hierarchical retrieval jump straight to "Termination"
# instead of scoring every section blind (spec section 7's worked example).
_INTENT_TOPICS = {
    "terminat": ["termination", "termination for cause", "termination for convenience"],
    "cancel": ["termination", "termination for convenience"],
    "pay": ["payment", "fees", "invoice"],
    "invoice": ["payment", "disputed invoices"],
    "fee": ["fees", "payment"],
    "confidential": ["confidentiality"],
    "liab": ["limitation of liability"],
    "indemn": ["indemnification", "insurance"],
    "insur": ["insurance"],
    "notice": ["notices", "notice requirements"],
    "compar": ["comparison"],
}

_COMPARISON_MARKERS = ("compare", "versus", " vs ", "difference between", "differ")
_TEMPORAL_MARKERS = ("days", "months", "years", "deadline", "period", "notice period", "when")


@dataclass
class QueryAnalysis:
    raw_query: str
    keywords: list[str] = field(default_factory=list)
    explicit_section_refs: list[str] = field(default_factory=list)
    candidate_topics: list[str] = field(default_factory=list)
    is_comparison: bool = False
    is_temporal: bool = False
    is_multi_hop: bool = False  # heuristic: multiple distinct candidate topics implied


def analyze_query(query: str) -> QueryAnalysis:
    lower = query.lower()

    words = re.findall(r"[a-zA-Z']+", lower)
    keywords = [w for w in words if w not in _STOPWORDS and len(w) > 2]

    explicit_refs = sorted(set(re.findall(r"section\s+(" + NUMBER_TOKEN_RE.pattern + r")", lower)))

    topics: list[str] = []
    for stem, topic_list in _INTENT_TOPICS.items():
        if stem in lower:
            for t in topic_list:
                if t not in topics:
                    topics.append(t)

    is_comparison = any(m in lower for m in _COMPARISON_MARKERS)
    is_temporal = any(m in lower for m in _TEMPORAL_MARKERS)
    # crude multi-hop signal: two+ distinct intent stems fired, or an explicit "and"
    # joining two different requirements (e.g. "termination rights ... and notice period")
    distinct_stems = [s for s in _INTENT_TOPICS if s in lower]
    is_multi_hop = len(distinct_stems) >= 2 or (" and " in lower and len(topics) >= 2)

    return QueryAnalysis(
        raw_query=query,
        keywords=keywords,
        explicit_section_refs=explicit_refs,
        candidate_topics=topics,
        is_comparison=is_comparison,
        is_temporal=is_temporal,
        is_multi_hop=is_multi_hop,
    )
