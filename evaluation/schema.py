"""
One question format used by the evaluation framework, generic enough to work
against ANY document — not just the synthetic sample — since `expected_keywords`
checks actual chunk *content* rather than section numbers that only this
sample document happens to have. Point scripts/evaluate.py at your own PDF +
your own questions.json in this shape.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class EvalQuestion(BaseModel):
    id: str
    question: str
    category: str  # simple_factual | section_specific | hierarchical | cross_reference |
                    # multi_hop | comparison | exception_condition | table | no_answer | ambiguous
    # A chunk counts as "relevant" ground truth if it contains ALL keywords in
    # ANY ONE of these groups (OR across groups, AND within a group). Most
    # questions need only one group (one supporting chunk); comparison and
    # some multi-hop questions genuinely have two+ equally-valid supporting
    # chunks (e.g. either side of a comparison), hence the group list rather
    # than a single flat list.
    expected_keyword_groups: list[list[str]] = Field(default_factory=list)
    is_answerable: bool = True   # False for "no_answer" category questions
    # OPTIONAL: your own reference answer, written by hand. It is NOT used for
    # scoring (scoring uses expected_keyword_groups) — it's stored next to the
    # model's answer in the saved report/CSV so you can compare the two by eye.
    expected_answer: str = ""
    notes: str = ""

    @property
    def expected_keywords(self) -> list[str]:
        """Convenience accessor: flattened keywords, used only for display/logging."""
        return [kw for group in self.expected_keyword_groups for kw in group]


class EvalDataset(BaseModel):
    document_name: str
    questions: list[EvalQuestion]