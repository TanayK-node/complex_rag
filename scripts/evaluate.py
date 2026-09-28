#!/usr/bin/env python3
"""
Usage:
    python3 scripts/evaluate.py path/to/document.pdf evaluation/sample_questions.json
    python3 scripts/evaluate.py path/to/document.pdf my_questions.json --out evaluation/results/run1.json

Runs the document through the same pipeline used for the synthetic sample:
1. Baseline retrieval comparison (spec section 21): 5 baselines x every
   question with expected_keyword_groups, reporting Precision@5, Hit@5 (the
   recall analog), MRR, NDCG@5, mean latency, and mean tokens retrieved.
2. Full-pipeline RAG evaluation (spec section 20): context precision/recall,
   answer relevance, faithfulness, answer correctness, and answerability
   accuracy (did it correctly recognize answerable vs. not), broken down by
   question category.

questions.json follows evaluation/schema.py's EvalDataset shape — write your
own for your own documents; evaluation/sample_questions.json is the one built
for the synthetic Master Service Agreement used throughout development.
"""
import argparse
import csv
import datetime
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.schema import EvalDataset
from evaluation.run_evaluation import run_baseline_comparison, run_rag_evaluation
from generation.providers import get_llm_provider, describe_provider, AnthropicProvider, GeminiProvider


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline against a document + question set.")
    parser.add_argument("pdf_path")
    parser.add_argument("questions_path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default=None,
                         help="Path for the JSON report. A .csv (open in Excel) and a .md (readable "
                              "question-by-question review) are written next to it. If omitted, a "
                              "timestamped file under evaluation/results/ is used automatically.")
    parser.add_argument("--skip-generation", action="store_true",
                         help="Only run the retrieval baseline comparison, skip RAG-level generation metrics "
                              "(faster; useful if you don't have an LLM configured and don't want the mock "
                              "provider's answers counted as representative)")
    args = parser.parse_args()

    with open(args.questions_path) as f:
        dataset = EvalDataset.model_validate_json(f.read())
    print(f"Loaded {len(dataset.questions)} questions for '{dataset.document_name}'")

    print("\n" + "=" * 78)
    print("BASELINE RETRIEVAL COMPARISON (spec section 21)")
    print("=" * 78)
    baseline_results = run_baseline_comparison(args.pdf_path, dataset, top_k=args.top_k)
    scored_n = baseline_results[0].n_questions
    print(f"(scored on {scored_n} of {len(dataset.questions)} questions — "
          f"those with expected_keyword_groups; 'no_answer' questions are excluded here)\n")
    _print_baseline_table(baseline_results)

    rag_result = None
    llm = None
    if not args.skip_generation:
        llm = get_llm_provider()
        provider_label = describe_provider(llm)
        print("\n" + "=" * 78)
        print(f"FULL-PIPELINE RAG EVALUATION (spec section 20) — LLM provider: {provider_label}")
        print("=" * 78)
        if not isinstance(llm, (AnthropicProvider, GeminiProvider)):
            print("NOTE: no LLM API key configured, so answer-quality numbers below reflect the\n"
                  "deterministic mock provider, not real generation quality. Retrieval/context\n"
                  "metrics above and context_precision/context_recall below are still meaningful;\n"
                  "answer_relevance/faithfulness/answer_correctness are not representative of a\n"
                  "real model until you set ANTHROPIC_API_KEY and re-run.\n")
        rag_result = run_rag_evaluation(args.pdf_path, dataset, llm)
        _print_rag_table(rag_result)
        _print_refusal_diagnostics(rag_result)

    out_path = args.out
    if out_path is None and rag_result is not None:
        slug = re.sub(r"[^a-z0-9]+", "-", dataset.document_name.lower()).strip("-")
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join("evaluation", "results", f"{slug}_{stamp}.json")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        report = {
            "document": dataset.document_name,
            "n_questions": len(dataset.questions),
            "llm_provider": describe_provider(llm) if rag_result else None,
            "baselines": [vars(b) for b in baseline_results],
            "rag_evaluation": {
                "overall": rag_result.overall,
                "by_category": rag_result.by_category,
                "rows": [vars(r) for r in rag_result.rows],
            } if rag_result else None,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nFull report saved to {out_path}")
        if rag_result:
            base = os.path.splitext(out_path)[0]
            _write_csv(base + ".csv", rag_result.rows)
            _write_markdown(base + ".md", dataset.document_name, describe_provider(llm), rag_result)
            print(f"Question-by-question review: {base}.csv (Excel)  and  {base}.md (readable)")


def _print_refusal_diagnostics(rag_result):
    o = rag_result.overall
    answerable = [r for r in rag_result.rows if r.is_answerable]
    print("\nREFUSAL DIAGNOSTICS (answerable questions only)")
    print(f"  answerable questions        : {len(answerable)}")
    print(f"  wrongly refused             : {int(o['wrongly_refused'])}  "
          f"(blocked by relevance gate: {int(o['wrongly_refused_by_gate'])}, "
          f"refused by the LLM: {int(o['wrongly_refused_by_llm'])})")
    print(f"  correctness when it answered: {o['correctness_when_answered']:.2f}   "
          f"<- the fair quality number; refusals score 0 in the table above")
    for r in answerable:
        if r.answer_source != "answered":
            print(f"    - {r.question_id} [{r.category}] {r.answer_source}, top_rerank_score={r.top_rerank_score:.2f}")


_CSV_COLUMNS = [
    "question_id", "category", "question", "expected_answer", "reference_passage", "answer",
    "answer_source", "is_answerable", "correctly_handled_answerability", "answer_correctness",
    "context_precision", "context_recall", "answer_relevance", "faithfulness",
    "top_rerank_score", "citations", "context_sections", "context_tokens", "input_tokens",
    "output_tokens", "latency_s", "groundedness_issues", "used_mock_fallback", "expected_keywords", "notes",
]


def _write_csv(path, rows):
    # utf-8-sig so Excel on Windows opens accents/curly quotes correctly
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            d = dict(vars(r))
            for k in ("citations", "context_sections", "expected_keywords"):
                d[k] = " | ".join(d[k])
            for k in ("answer", "reference_passage", "expected_answer"):
                d[k] = d[k].replace("\r", " ")
            w.writerow(d)


def _write_markdown(path, doc_name, provider, rag_result):
    o = rag_result.overall
    lines = [f"# Evaluation review — {doc_name}", "",
             f"LLM provider: **{provider}**  ", f"Questions: {int(o['n'])}  ",
             f"Wrongly refused (answerable but no answer): **{int(o['wrongly_refused'])}** "
             f"(gate {int(o['wrongly_refused_by_gate'])}, LLM {int(o['wrongly_refused_by_llm'])})  ",
             f"Correctness when it answered: **{o['correctness_when_answered']:.2f}**", ""]
    for r in rag_result.rows:
        flag = {"answered": "answered", "gate_refusal": "REFUSED by relevance gate",
                "llm_refusal": "REFUSED by LLM"}.get(r.answer_source, r.answer_source)
        lines += [f"## {r.question_id} · {r.category} · {flag}", "",
                  f"**Question:** {r.question}", ""]
        if r.expected_answer:
            lines += [f"**Your expected answer:** {r.expected_answer}", ""]
        if r.reference_passage:
            lines += ["**Source passage in the document:**", "", "> " + r.reference_passage.replace("\n", "\n> "), ""]
        else:
            lines += ["**Source passage in the document:** _(none — question is meant to be unanswerable)_" if not r.is_answerable
                      else "**Source passage in the document:** _(not found)_", ""]
        lines += [f"**System answer:** {r.answer}", "",
                  f"- correctness (keyword proxy): {r.answer_correctness:.2f} · relevance: {r.answer_relevance:.2f} · "
                  f"faithfulness: {r.faithfulness:.0f} · ctx precision/recall: {r.context_precision:.2f}/{r.context_recall:.2f}",
                  f"- top rerank score: {r.top_rerank_score:.2f} · context: {r.context_tokens} tok · "
                  f"LLM tokens in/out: {r.input_tokens}/{r.output_tokens} · latency: {r.latency_s}s",
                  f"- sent to LLM: {', '.join(r.context_sections) or '(nothing)'}"]
        if r.groundedness_issues:
            lines.append(f"- groundedness issues: {r.groundedness_issues}")
        if r.notes:
            lines.append(f"- note: {r.notes}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _print_baseline_table(results):
    header = f"{'Baseline':38} {'P@5':>6} {'Hit@5':>7} {'MRR':>6} {'NDCG@5':>7} {'Lat(ms)':>8} {'Tokens':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.baseline_id}. {r.name:35} {r.mean_precision_at_5:6.2f} {r.mean_hit_at_5:7.2f} "
              f"{r.mean_mrr:6.2f} {r.mean_ndcg_at_5:7.2f} {r.mean_latency_ms:8.1f} {r.mean_tokens_retrieved:7.0f}")


def _print_rag_table(rag_result):
    print(f"{'Category':20} {'CtxPrec':>8} {'CtxRec':>7} {'AnsRel':>7} {'Faith':>6} {'Correct':>8} {'Answ.Acc':>9} {'n':>3}")
    for cat, m in rag_result.by_category.items():
        print(f"{cat:20} {m['context_precision']:8.2f} {m['context_recall']:7.2f} {m['answer_relevance']:7.2f} "
              f"{m['faithfulness']:6.2f} {m['answer_correctness']:8.2f} {m['answerability_accuracy']:9.2f} {m['n']:3.0f}")
    m = rag_result.overall
    print("-" * 78)
    print(f"{'OVERALL':20} {m['context_precision']:8.2f} {m['context_recall']:7.2f} {m['answer_relevance']:7.2f} "
          f"{m['faithfulness']:6.2f} {m['answer_correctness']:8.2f} {m['answerability_accuracy']:9.2f} {m['n']:3.0f}")


if __name__ == "__main__":
    main()