# Hierarchical Hybrid RAG — Legal/Enterprise Documents

Status: **Phases 1-6 complete and tested**: ingestion, hybrid+hierarchical
retrieval, reranking, context building, grounded generation with citations,
groundedness verification, AND a working evaluation framework with a
5-baseline ablation comparison. Phases 7-8 (input-sanitization hardening,
FastAPI backend, UI) are designed but not yet implemented — you don't need
them to use this: `scripts/` gives you a CLI for everything.

## Quick start — use this on your own documents right now

```bash
pip install -r requirements.txt

# 1. Ingest and index a PDF (yours, or the bundled synthetic one) — do this once per document
python3 scripts/index_document.py path/to/your.pdf --name "My Contract"
#   -> saved to data/processed/my-contract/

# 2. Ask it questions interactively
python3 scripts/ask.py data/processed/my-contract --verbose

# 3. Evaluate it against a list of questions you write (see evaluation/schema.py
#    for the format; evaluation/sample_questions.json is a worked example)
python3 scripts/evaluate.py path/to/your.pdf my_questions.json --out evaluation/results/run1.json
```

Set `ANTHROPIC_API_KEY` (or `RAG_LLM_API_KEY`) in your environment before
running `ask.py`/`evaluate.py` to get real Claude-generated answers instead
of the deterministic mock provider (see "Using a real LLM" below) — no code
changes needed either way, `get_llm_provider()` picks it up automatically.

## What works right now

```
PDF → pdf_parser → structure_detector → tree_builder → chunker → summarizer   [Phase 1]
                                                                    ↓
query → query_analysis ─┬→ vector_store (TF-IDF/SVD + FAISS)  ─┐
                         ├→ lexical_search (BM25)               ├→ hybrid_retriever  [Phase 2]
                         └→ hierarchical_retriever (tree walk)  ┘   (fuse + dedupe +
                                                                      cross-ref expand)
                                                                    ↓
                                                      reranker (feature-based        [Phase 3]
                                                      cross-encoder-style scorer)
                                                                    ↓
                                       [below relevance threshold?] → INSUFFICIENT EVIDENCE
                                                                    ↓ no
                                                      context_builder (group, order,   [Phase 3]
                                                      dedupe, token-budget trim, cite)
                                                                    ↓
                                                      prompt_builder → LLM provider     [Phase 4]
                                                      (Anthropic API / mock) → answer
                                                                    ↓
                                                      groundedness check → retry loop   [Phase 4]
                                                      (regenerate, then widen context,
                                                      then give up honestly)

scripts/index_document.py + ingestion/persistence.py  → save/load a document's full index [Phase 5]
scripts/ask.py                                        → interactive Q&A CLI              [Phase 5]
scripts/evaluate.py + evaluation/*                    → 5-baseline ablation + RAG metrics [Phase 6]
```

Run the smoke tests (regenerates the synthetic contract if needed):

```bash
python3 data/sample_docs/generate_sample.py   # only needed once
python3 tests/test_ingestion.py               # Phase 1
python3 tests/test_retrieval.py               # Phase 2
python3 tests/test_context.py                 # Phase 3
python3 tests/test_generation.py              # Phase 4
python3 tests/test_evaluation.py              # Phase 5/6
```

## A real result from `test_evaluation.py` (not fabricated — rerun it yourself)

This is the whole architectural thesis of the project, actually measured on
the synthetic contract:

| Baseline | Hit@5 | Tokens retrieved |
|---|---|---|
| 1. Fixed-size chunking, dense-only | 0.87 | 1106 |
| 2. Structure-aware chunking, dense-only | **1.00** | **237** |

Structure-aware chunking finds the right passage *more* reliably while
retrieving **~4.7x fewer tokens** — exactly the token-optimization argument
spec section 19 is asking the ablation study to actually prove, not assert.
Full 5-baseline table (including hybrid/hierarchy/reranking) is printed by
`scripts/evaluate.py` and saved to whatever `--out` path you give it.

## Evaluating your own documents

1. Write your own `evaluation/schema.py`-shaped questions JSON. The key
   design choice: relevance ground truth is `expected_keyword_groups` — a
   list of keyword groups, where a retrieved chunk counts as "relevant" if it
   contains **all** keywords in **any one** group. Most questions need one
   group (one supporting passage); comparison/some multi-hop questions
   legitimately have two (either side of the comparison).
2. `no_answer`-category questions should have `"expected_keyword_groups": []`
   and `"is_answerable": false` — these are excluded from the retrieval
   baseline table (nothing to score) but ARE checked in the RAG evaluation's
   `answerability_accuracy` column.
3. Run `scripts/evaluate.py your.pdf your_questions.json`.

## Using a real LLM instead of the mock

By default `generation/providers.get_llm_provider()` falls back to
`DeterministicMockProvider` because no API key is configured in this sandbox.
To use real Claude generation:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

`AnthropicProvider` is a real, complete implementation against
`api.anthropic.com/v1/messages` — **it has not been live-tested in this
environment** (no key was available here). Test it yourself with a real key
before relying on it; don't take "the code is written" as "it's been proven
correct" for this one class.

## Folder structure

```
ingestion/       PDF parsing, structure detection, chunking, persistence (DONE — Phase 1, 5)
hierarchy/       Tree builder, summarizer, hierarchical retriever (DONE — Phase 1+2)
schemas/         Pydantic contracts shared by every stage (DONE)
config/          Central settings (DONE)
vector_store/    Embedder interface + FAISS wrapper (DONE — Phase 2)
lexical_search/  BM25 index wrapper (DONE — Phase 2)
retrieval/       Query analysis + hybrid fusion (DONE — Phase 2)
reranking/       Reranker interface + heuristic cross-encoder-style scorer (DONE — Phase 3)
context/         Context builder + citation formatting (DONE — Phase 3)
generation/      LLM provider interface, prompt builder, full query pipeline (DONE — Phase 4)
guardrails/      Groundedness verification (DONE — Phase 4). Prompt-injection
                 defense is in the prompt itself (spec section 24); a dedicated
                 input-sanitization layer is still Phase 7's job.
evaluation/      Eval schema + dataset, retrieval/RAG metrics, 5-baseline
                 comparison, sample_questions.json (DONE — Phase 5/6)
scripts/         index_document.py, ask.py, evaluate.py — the CLI (DONE — Phase 5/6)
api/             FastAPI backend (NOT YET BUILT — Phase 7)
ui/              Streamlit/React UI (NOT YET BUILT — Phase 7)
```

## Choosing an LLM provider — speed vs. accuracy vs. cost

If your mentor wants **seconds, not a minute**, per query, here's the actual
trade-off, not just "use a bigger model":

**Local open-source model (Llama, Mistral, Qwen, etc.):**
- No per-query API cost, full data privacy (real advantage for confidential
  legal/financial documents), no external rate limits.
- But: **CPU-only inference is not fast.** A 7-8B parameter model on CPU
  typically takes several seconds to tens of seconds even for a short answer.
  "Local" does not mean "fast" unless you have a real GPU. A quantized 7-8B
  model on a consumer GPU (RTX 3090/4090-class) can realistically hit
  1-3 seconds for a short grounded answer — but that requires owning or
  renting that GPU, plus running and maintaining an inference server
  (Ollama, vLLM, llama.cpp), which is real extra engineering surface for a
  project whose actual contribution is the retrieval architecture, not the
  generation model.

**Hosted "fast-tier" API model (Claude Haiku, Gemini Flash, GPT-4o-mini, etc.):**
- Consistently fast (typically 1-3s for a short completion), zero infra to
  run, strong quality even at the cheap/fast tier — because in a RAG system,
  **retrieval quality does most of the work.** A cheap, fast model reading a
  well-assembled, short, structure-aware context (this project's whole point)
  answers well *because the context is good*, not because the model is huge.
- Cost is real but small for short completions; data leaves your machine
  (check the provider's data-use policy for your document sensitivity level
  — Anthropic and OpenAI's API tiers don't train on your data by default;
  Google's **free** Gemini tier has looser terms and may use your data to
  improve their products — worth knowing before sending real client
  documents through it).

**My recommendation for "fast, in seconds, for a document Q&A system":** a
hosted fast-tier model (Claude Haiku or Gemini Flash) over a local model,
specifically because the local-model route only gets you to "seconds" with a
GPU you'd need to provision and maintain — an ops cost this project doesn't
otherwise need. Both `AnthropicProvider` and `GeminiProvider` are implemented
(`generation/providers.py`) behind the same `LLMProvider` interface, so
switching is one environment variable, not a code change — you can actually
run the same eval harness against both and report real speed/cost/quality
numbers for your mentor, which is a more convincing answer than an assertion.

**On free-tier Gemini rate limits specifically — yes, this is a real
problem, not a hypothetical one.** Free-tier API keys have low per-minute
request quotas that you *will* hit the moment you run `evaluate.py` (which
fires one LLM call per question, back-to-back — 30-50 calls in quick
succession). Two things are already built in to help:
- Both `AnthropicProvider` and `GeminiProvider` now retry on HTTP 429
  (rate-limited) and 503 (overloaded) with exponential backoff (1s → 2s → 4s
  → 8s) before failing — see `_post_with_retry()` in
  `generation/providers.py`. This won't make free-tier limits disappear, but
  it means a transient rate-limit doesn't crash a 40-question eval run.
- For a live demo in front of your mentor specifically, Claude Haiku's
  pay-as-you-go tier (a few dollars of credit) is the more reliable choice —
  free tiers are for casual testing, not for a timed demo where a 429 mid-answer
  would look bad. Check current published rate limits for whichever
  provider/tier you pick — they change often enough that I'd verify on the
  provider's own docs before your demo, not from what's written here.

**Neither `AnthropicProvider` nor `GeminiProvider` has been live-tested in
this sandbox** — its network egress allowlist blocks
`generativelanguage.googleapis.com` outright (confirmed: the request reaches
the proxy and gets an explicit rejection, not a timeout), and there's no
Anthropic API key configured here either. Test both with real keys in your
own environment before a demo.

## Known real-world stress-test findings

Tested against two real documents you provided (not the synthetic sample):
a 186-page UK government "Short Form Contract" (converted from `.odt` via
LibreOffice headless) and a 117-page Everus Construction Group 10-K annual
report. Both are now bundled under `data/sample_docs/real_world/`, indexed
under `data/processed/`, with eval question sets in
`evaluation/real_world_contract_questions.json` and
`evaluation/real_world_10k_questions.json`. Real, run, non-fabricated
findings from those runs:

- **Structure detection degrades significantly on documents that don't use
  clean decimal numbering.** The UK contract mixes Roman-numeral top-level
  sections (I., II., III.) with decimal sub-clauses (4.1, 11.3) — only 20
  tree nodes were detected for 186 pages, and several top-level Roman
  numeral sections were missed entirely. The 10-K uses "Item 1", "PART I",
  "Note 1" conventions the heading regex has no pattern for at all. Two
  real bugs were caught and fixed by testing against these documents (see
  `structure_detector._looks_like_heading`): bold financial-table column
  headers like "2025 2024" and pull-quote figures like "$3.75B" were being
  misclassified as section headings before the fix.
- **On the UK contract, hierarchical retrieval measurably *hurt* Hit@5**
  (0.78 vs. 0.89 for the other four baselines) — direct evidence that
  hierarchy only helps when the tree it's built on is a decent
  representation of the document's real structure. On the (differently
  messy) 10-K, hybrid (dense+BM25) retrieval clearly helped over dense-only
  (Hit@5 0.75 vs 0.62), and reranking improved MRR most (0.67 vs 0.57-0.60).
  These are two different documents with two different failure modes,
  reported as measured, not smoothed into one "it works" number.
- **Cross-reference extraction found essentially nothing in the UK
  contract**, because its source document uses hyperlinked cross-references
  whose *visible text is the target clause's title* ("in accordance with
  clause Data Protection and Security"), not "Section 14.8" — a pattern
  `CROSS_REF_RE` has no way to catch, since it looks for "Section <number>".
  This is a different, and arguably harder, cross-reference convention than
  the spec's own example assumes.
- **pdfplumber's default table detection over-fires on text-heavy PDFs
  without visible gridlines** — 75 "tables" were detected in the 10-K,
  many of which are actually justified paragraph text pdfplumber's
  line-detection heuristic mistook for table structure. Not fixed here;
  would need `table_settings` tuned toward an explicit-gridline detection
  strategy for financial filings specifically.

None of this is hidden in the numbers `scripts/evaluate.py` prints — run
`python3 scripts/evaluate.py data/sample_docs/real_world/annual_report_10k.pdf evaluation/real_world_10k_questions.json`
yourself to see it.

## Real-world documents referenced in this doc

`data/sample_docs/real_world/` holds the converted/copied PDFs used for the
stress test above (the UK contract as converted PDF, the 10-K, and a GDPR
excerpt — see below). **The 186-page contract and 117-page 10-K PDFs are
excluded from the packaged download** since you already have those source
files — drop them back into that folder (or point `scripts/index_document.py`
directly at your originals) to reproduce:

```bash
python3 scripts/index_document.py your_10k.pdf --name "Everus 2025 10-K"
python3 scripts/evaluate.py your_10k.pdf evaluation/real_world_10k_questions.json
```

(`evaluation/real_world_contract_questions.json`,
`evaluation/real_world_10k_questions.json`, and
`evaluation/real_world_gdpr_questions.json` — the question sets used above —
ARE included, along with `data/sample_docs/real_world/gdpr_excerpt.pdf`
itself, since it's small and generated by this project.)

### Third stress-test document: GDPR excerpt (Regulation (EU) 2016/679)

Added specifically for a third, genuinely different numbering convention:
`Chapter I` / `Article 4` / paragraph `1.` / point `(a)` — neither the UK
contract's decimal numbering nor the 10-K's "Item 1A" style. Official,
publicly-reusable EU legislative text (source: EUR-Lex / gdpr-info.eu),
7 Articles across 4 Chapters (`data/sample_docs/real_world/generate_gdpr_excerpt.py`
regenerates it). Two more real bugs surfaced and fixed while building this:

- **The heading detector had no pattern for "Article N" at all** — a
  title-case, word-prefixed convention distinct from both decimal numbers
  and ALL-CAPS headings. Added `ARTICLE_HEADING_RE` in `structure_detector.py`.
- **The header/footer stripping heuristic was a false-positive risk for any
  short, recurring heading.** Every "Article N" line normalizes to the same
  masked key ("Article #") and recurs on most pages — which is exactly what
  the *page-count* check for headers/footers was looking for, so it was
  stripping every Article heading as if it were a running footer. Fixed by
  adding a vertical-position check: a real header/footer sits at the same
  y-position on every page; a heading's y-position varies with page content.
  See `pdf_parser._strip_headers_footers`.

On this document (small, and its Chapter→Article tree comes out clean),
hierarchical retrieval clearly **helps** (MRR jumps from 0.86 to 0.97 vs.
hybrid-without-hierarchy) — a third, different data point completing the
picture from the UK contract (hierarchy hurt) and the 10-K (hierarchy
neutral): hierarchy's value tracks how clean the underlying tree is, not
some property of "hierarchical retrieval" in the abstract.

## Known limitations of Phase 1

- **Heading detection is heuristic** (font size + boldness + numbering regex),
  not a learned layout model. It works well on documents with consistent
  numbering/typography (the common case for contracts and policies), but a
  document that numbers headings and body clauses in the exact same font
  will need the thresholds in `structure_detector.py` tuned, or a swap to a
  layout-model-based detector — the module boundary is designed for that swap.
- **Token counts are word-count approximations** (`chunker._approx_tokens`),
  not a real tokenizer. Fine for chunk-size budgeting; swap in `tiktoken` (or
  the provider's own tokenizer) before reporting exact token usage in Section 19.
- **Summaries are extractive** (child section titles + text lead-ins), not
  LLM-generated. This was a deliberate Phase 1 choice to keep ingestion
  dependency-free and deterministic; `hierarchy/summarizer.py` is the single
  seam to swap in an LLM call later.
- **Only tested on one synthetic document.** It exercises nested numbering,
  cross-references, a table, and multi-page clauses, but has not been run
  against a real, messy, scanned, or OCR'd legal PDF yet — that's the first
  thing to validate once you provide a real document.
- **Dense retrieval uses TF-IDF+SVD, not a transformer embedding model.**
  This environment doesn't have the disk headroom to install `torch` +
  download `sentence-transformers` weights (confirmed: installing it filled
  the disk). `vector_store/embedder.py` defines an `Embedder` interface with
  `TfidfEmbedder` (active) and a ready-to-go `SentenceTransformerEmbedder`
  (inactive) — switching is a one-line change in `get_embedder()` once you're
  running this somewhere with network/disk room for it. TF-IDF+SVD will
  under-perform a real embedding model specifically on paraphrased queries
  that share little vocabulary with the source text — worth re-running the
  eval suite (Phase 6) once the swap is made.
- **Hybrid fusion is min-max normalization + weighted average**, not
  reciprocal rank fusion or a learned fusion model. Simple and explainable;
  Phase 6's baseline comparison is the place to check whether a fancier
  fusion actually earns its complexity.
- **Cross-reference expansion is naive**: one hop, fixed low score, no check
  for whether the referenced section is actually relevant to the *query*
  (only that it was referenced by something relevant). This is deliberately
  left for the reranker (Phase 3) to sort out, rather than guessing here.
- **Reranker is feature-based, not a real cross-encoder.** Same disk
  constraint as the embedder: a cross-encoder model is another transformer
  download this sandbox can't fit. `reranking/heuristic_reranker.py` scores
  query-chunk relevance via phrase overlap, term recall/precision, and
  section-title match — a genuinely different (and stronger) signal than the
  retrieval-stage fused score, but not as good as a trained cross-encoder at
  catching subtle semantic relevance. `CrossEncoderReranker` and
  `LLMReranker` are defined and ready to swap in via `get_reranker()`.
- **Context grouping merges by node**, so a table chunk and a paragraph
  chunk under the same section share one header — correct for this test
  document, but if a node ever accumulates chunks in a very different order
  than reading order, the "\n\n".join may not read as smoothly as
  hand-written prose would.
- **`AnthropicProvider` is untested in this environment** (no API key here —
  see "Using a real LLM instead of the mock" above). The pipeline around it
  (prompting, groundedness checking, retries) IS tested, via a mock provider
  and a synthetic "flaky" one that deliberately fails groundedness once.
- **The pre-generation relevance threshold (`similarity_threshold=0.28`) is a
  hand-picked starting point**, validated against exactly two representative
  queries (one clearly answerable, one clearly not). It's a real engineering
  lever — not a magic constant — and the right place to tune it is Phase 6's
  evaluation framework, against a labeled set of answerable/unanswerable
  questions, not by eyeballing two examples.
- **Groundedness checking is lexical-overlap-based**, not an NLI/entailment
  model. It reliably catches fabricated citations and fabricated numbers
  (both high-stakes in legal text), but would miss a subtler unsupported
  inference phrased entirely in words that already appear in context.
- **The pre-generation relevance threshold has a real, observed failure
  mode.** Rephrasing "Which law firm represents the Vendor in litigation?"
  (correctly blocked) to "Which law firm represents the Vendor?" (NOT
  blocked — scored 0.293, just above the 0.28 threshold) changes the outcome,
  because a common word ("Vendor") appearing throughout the document adds
  enough lexical noise to a heuristic scorer to tip a genuinely unanswerable
  question over threshold. This isn't hidden — it's exactly what
  `answerability_accuracy < 1.0` on the `no_answer` category in the eval
  report reflects. A real embedding/cross-encoder model would very likely
  handle this better; this is one of the concrete, measured cases where the
  disk-constrained heuristic stand-ins cost real accuracy, not just "would
  probably be better in theory."
- **Evaluation ground truth (`expected_keyword_groups`) is written by hand**,
  not derived from the document automatically. Writing good keyword groups
  for your own real documents will take real effort — pick short, specific
  phrases unlikely to appear together by coincidence elsewhere.
- **`fixed_size_chunks` (Baseline 1) ignores tables entirely** — it splits
  pdfplumber's line-level text only, the same source the structure-aware
  chunker excludes table regions from. This is realistic (a truly naive
  chunker usually does the same), but it means Baseline 1's poor performance
  on table questions is partly "naive chunking doesn't have a table
  strategy at all," not purely "worse chunk boundaries."
- **Baseline comparison reports one aggregate score per baseline**, not a
  per-category breakdown (the RAG-level evaluation IS broken down by
  category — see the table `scripts/evaluate.py` prints). Category-level
  baseline breakdown (e.g. "does hierarchy help more on `multi_hop`
  questions specifically?") would be a natural next addition to
  `evaluation/run_evaluation.py` if you want to dig into *why* one baseline
  wins, not just that it does.
