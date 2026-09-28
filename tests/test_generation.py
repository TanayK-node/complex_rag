import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pipeline import ingest_pdf
from vector_store.embedder import get_embedder
from vector_store.faiss_index import VectorStore
from lexical_search.bm25_index import LexicalIndex
from reranking.heuristic_reranker import get_reranker
from generation.providers import DeterministicMockProvider
from generation.llm_provider import LLMProvider, LLMResponse
from generation.pipeline import answer_query
from guardrails.groundedness import check_groundedness
from context.context_builder import build_context, ContextBlock, ContextBundle

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs", "master_service_agreement.pdf")


class FlakyProvider(LLMProvider):
    """First call returns a deliberately ungrounded answer (bad citation +
    fabricated figure); second call delegates to the real mock provider.
    Exists purely to exercise the retry-on-failed-groundedness path."""

    def __init__(self):
        self.calls = 0
        self.mock = DeterministicMockProvider()

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text="The termination fee is $99,999 and thirty days notice is required "
                     "[Source: Master Service Agreement, Section 99.9].",
                model="flaky-mock",
            )
        return self.mock.complete(system, user, max_tokens)


def main():
    tree, chunks = ingest_pdf(SAMPLE_PDF, document_name="Master Service Agreement")
    chunks_by_id = {c.chunk_id: c for c in chunks}

    vector_store = VectorStore(get_embedder())
    vector_store.build(chunks)
    lexical_index = LexicalIndex()
    lexical_index.build(chunks)
    reranker = get_reranker()
    mock_llm = DeterministicMockProvider()

    print("=" * 78)
    print("TEST 1: answerable question")
    print("=" * 78)
    result = answer_query(
        "How many days' notice must Client give to terminate for convenience?",
        tree, chunks_by_id, vector_store, lexical_index, reranker, mock_llm,
    )
    print(f"ANSWER: {result.answer}")
    print(f"CITATIONS: {result.citations}")
    print(f"GROUNDEDNESS: passed={result.groundedness.passed} retries={result.retries_used}")
    assert result.groundedness.passed, "answerable question should pass groundedness on first try"
    assert result.citations, "a grounded, non-insufficient answer should carry at least one citation"

    print()
    print("=" * 78)
    print("TEST 2: unanswerable question -> insufficient evidence")
    print("=" * 78)
    result2 = answer_query(
        "Which law firm represents the Vendor in litigation?",
        tree, chunks_by_id, vector_store, lexical_index, reranker, mock_llm,
    )
    print(f"ANSWER: {result2.answer}")
    assert result2.answer.upper().startswith("INSUFFICIENT EVIDENCE"), \
        "question with no basis in the document should be flagged insufficient evidence"
    assert result2.groundedness.is_insufficient_evidence

    print()
    print("=" * 78)
    print("TEST 3: groundedness checker directly catches a fabricated answer")
    print("=" * 78)
    fake_context = ContextBundle(
        blocks=[ContextBlock(
            document_name="Master Service Agreement", section_number="4.3",
            section_title="Termination for Convenience", parent_section="Term and Termination",
            page_start=3, page_end=3,
            text="Client may terminate this Agreement for convenience upon sixty (60) days' prior written notice.",
            citation="[Source: Master Service Agreement, Section 4.3, Page 3]",
            chunk_ids=["fake-chunk-1"], max_rerank_score=1.0,
        )],
        context_text="[Document: Master Service Agreement | Section 4.3 (Termination for Convenience) | Page 3]\n"
                      "Client may terminate this Agreement for convenience upon sixty (60) days' prior written notice.",
        total_tokens=30, included_chunk_ids=["fake-chunk-1"], dropped_for_budget=[],
    )
    fabricated_answer = ("Client must pay a $99,999 penalty and give ninety days notice "
                          "[Source: Master Service Agreement, Section 99.9].")
    check = check_groundedness(fabricated_answer, fake_context)
    print(f"passed={check.passed} bad_citations={check.bad_citations} "
          f"unsupported_numbers={check.unsupported_numbers}")
    assert not check.passed, "fabricated citation + fabricated figure must fail groundedness"
    assert check.bad_citations, "Section 99.9 does not exist in context and must be flagged"
    assert any(n.replace(",", "").replace("$", "") == "99999" for n in check.unsupported_numbers), \
        f"expected the fabricated $99,999 figure to be flagged, got {check.unsupported_numbers}"

    print()
    print("=" * 78)
    print("TEST 4: pipeline actually retries when the first draft fails groundedness")
    print("=" * 78)
    flaky = FlakyProvider()
    result4 = answer_query(
        "How many days' notice must Client give to terminate for convenience?",
        tree, chunks_by_id, vector_store, lexical_index, reranker, flaky,
    )
    print(f"ANSWER: {result4.answer}")
    print(f"retries_used={result4.retries_used}, provider calls={flaky.calls}")
    assert flaky.calls >= 2, "pipeline should have retried after the first ungrounded draft"
    assert result4.retries_used >= 1

    print("\nAll Phase 4 smoke-test assertions passed.")


if __name__ == "__main__":
    main()
