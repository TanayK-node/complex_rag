import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.schema import EvalDataset
from evaluation.run_evaluation import run_baseline_comparison, run_rag_evaluation
from generation.providers import DeterministicMockProvider

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs", "master_service_agreement.pdf")
QUESTIONS = os.path.join(os.path.dirname(__file__), "..", "evaluation", "sample_questions.json")


def main():
    with open(QUESTIONS) as f:
        dataset = EvalDataset.model_validate_json(f.read())
    assert 30 <= len(dataset.questions) <= 50, "eval dataset should have 30-50 questions per spec section 20"
    categories = {q.category for q in dataset.questions}
    expected_categories = {
        "simple_factual", "section_specific", "hierarchical", "cross_reference",
        "multi_hop", "comparison", "exception_condition", "table", "no_answer", "ambiguous",
    }
    assert categories == expected_categories, f"missing categories: {expected_categories - categories}"

    print("Running baseline comparison...")
    results = run_baseline_comparison(SAMPLE_PDF, dataset, top_k=5)
    by_id = {r.baseline_id: r for r in results}

    # every metric must be a valid probability/rank score
    for r in results:
        for name, val in [("precision", r.mean_precision_at_5), ("hit", r.mean_hit_at_5),
                           ("mrr", r.mean_mrr), ("ndcg", r.mean_ndcg_at_5)]:
            assert 0.0 <= val <= 1.0 + 1e-9, f"baseline {r.baseline_id} {name}={val} out of [0,1]"

    # core thesis of the whole project: structure-aware chunking should retrieve
    # at least as reliably as naive fixed-size chunking, using far fewer tokens
    assert by_id[2].mean_hit_at_5 >= by_id[1].mean_hit_at_5, \
        "structure-aware chunking should not retrieve worse than fixed-size chunking"
    assert by_id[2].mean_tokens_retrieved < by_id[1].mean_tokens_retrieved, \
        "structure-aware chunks should be more token-efficient than fixed-size windows"

    # hybrid+hierarchy+rerank (5) should rank the right chunk earlier (or as early)
    # as dense-only (2) on average, per spec section 21's whole point
    assert by_id[5].mean_mrr >= by_id[2].mean_mrr - 0.05, \
        f"baseline 5 (full pipeline) MRR ({by_id[5].mean_mrr}) regressed badly vs dense-only ({by_id[2].mean_mrr})"

    print(f"  baseline 1 (fixed-size): hit@5={by_id[1].mean_hit_at_5:.2f}, tokens={by_id[1].mean_tokens_retrieved:.0f}")
    print(f"  baseline 2 (structure):  hit@5={by_id[2].mean_hit_at_5:.2f}, tokens={by_id[2].mean_tokens_retrieved:.0f}")
    print(f"  baseline 5 (full):       mrr={by_id[5].mean_mrr:.2f}, ndcg@5={by_id[5].mean_ndcg_at_5:.2f}")

    print("\nRunning RAG-level evaluation (mock provider)...")
    rag_result = run_rag_evaluation(SAMPLE_PDF, dataset, DeterministicMockProvider())
    assert 0.0 <= rag_result.overall["faithfulness"] <= 1.0
    assert 0.0 <= rag_result.overall["answerability_accuracy"] <= 1.0
    assert rag_result.overall["n"] == len(dataset.questions)
    # every category should have been scored at least once
    assert set(rag_result.by_category.keys()) == expected_categories

    print(f"  overall context_recall={rag_result.overall['context_recall']:.2f}, "
          f"faithfulness={rag_result.overall['faithfulness']:.2f}, "
          f"answerability_accuracy={rag_result.overall['answerability_accuracy']:.2f}")

    print("\nAll Phase 5/6 smoke-test assertions passed.")


if __name__ == "__main__":
    main()
