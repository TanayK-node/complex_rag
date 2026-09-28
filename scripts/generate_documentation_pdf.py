from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=20, spaceAfter=6)
subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=11, textColor=colors.HexColor("#555555"),
                                 alignment=TA_CENTER, spaceAfter=24)
h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, spaceBefore=18, spaceAfter=8,
                     textColor=colors.HexColor("#1a2b4c"))
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=6,
                     textColor=colors.HexColor("#2c3e50"))
body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, leading=14, spaceAfter=8)
small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8.5, leading=12, spaceAfter=6,
                        textColor=colors.HexColor("#444444"))
code = ParagraphStyle("Code", parent=styles["Normal"], fontName="Courier", fontSize=8.5, leading=12,
                       backColor=colors.HexColor("#f4f4f4"), borderPadding=6, spaceAfter=8)

story = []

# --- Title ---
story.append(Paragraph("Hierarchical Hybrid RAG for Legal &amp; Enterprise Documents", title_style))
story.append(Paragraph("Project documentation — Phases 1-6 (ingestion through evaluation)", subtitle_style))

# --- 1. Overview ---
story.append(Paragraph("1. What this is", h1))
story.append(Paragraph(
    "A retrieval-augmented generation (RAG) system purpose-built for complex legal and enterprise "
    "PDFs: nested clauses, cross-references, tables, and page boundaries. Unlike a basic "
    "chunk-and-embed pipeline, it preserves document hierarchy end to end and uses that structure "
    "during retrieval, reranking, and context assembly.", body))

story.append(Paragraph("Pipeline stages", h2))
stages = [
    "PDF parsing (layout-aware: font size, boldness, page, table geometry)",
    "Structure detection (headings vs. body text; cross-reference extraction)",
    "Hierarchical document tree (parent/child sections down to clause level)",
    "Structure-aware chunking (clauses never split mid-way unless oversized)",
    "Hybrid retrieval: dense (TF-IDF/SVD + FAISS) + lexical (BM25) + hierarchical tree walk",
    "Reranking (feature-based cross-encoder-style scorer)",
    "Relevance gate (blocks generation before it starts if nothing is relevant enough)",
    "Context building (grouped, ordered, deduplicated, token-budgeted, cited)",
    "Grounded generation (LLM call constrained to retrieved context only)",
    "Groundedness verification with a bounded retry loop",
    "Evaluation framework: 5-baseline ablation + RAG-level metrics",
]
story.append(ListFlowable([ListItem(Paragraph(s, body), leftIndent=10) for s in stages], bulletType="1"))

story.append(Paragraph("Folder layout", h2))
story.append(Paragraph(
    "ingestion/ · hierarchy/ · schemas/ · config/ · vector_store/ · lexical_search/ · retrieval/ · "
    "reranking/ · context/ · generation/ · guardrails/ · evaluation/ · scripts/ (CLI entry points). "
    "Not yet built: api/ and ui/ (a FastAPI backend and web UI are Phase 7 — everything below runs "
    "from the command line today).", body))

# --- 2. How to run ---
story.append(Paragraph("2. How to run it", h1))
story.append(Paragraph("2.1 Install dependencies (once)", h2))
story.append(Paragraph("pip install -r requirements.txt", code))

story.append(Paragraph("2.2 Index a document (once per document)", h2))
story.append(Paragraph(
    'python3 scripts/index_document.py path/to/your.pdf --name "My Document"<br/>'
    "→ saved to data/processed/&lt;slug&gt;/ (tree, chunks, dense index, lexical index)", code))

story.append(Paragraph("2.3 Ask it questions interactively", h2))
story.append(Paragraph("python3 scripts/ask.py data/processed/&lt;slug&gt; --verbose", code))

story.append(Paragraph("2.4 Evaluate it against a question set", h2))
story.append(Paragraph(
    "python3 scripts/evaluate.py path/to/your.pdf your_questions.json --out evaluation/results/run.json", code))
story.append(Paragraph(
    "Question sets follow evaluation/schema.py. The key field is <b>expected_keyword_groups</b>: a list "
    "of keyword groups, where a retrieved chunk counts as relevant if it contains every keyword in "
    "<i>any one</i> group. Most questions need one group (one supporting passage); comparison questions "
    "can have two (either side of the comparison). \u201cno_answer\u201d questions use an empty list and "
    "is_answerable: false.", body))

story.append(Paragraph("2.5 Set an LLM provider (optional — falls back to a deterministic mock otherwise)", h2))
story.append(Paragraph("export ANTHROPIC_API_KEY=sk-ant-...      # or\nexport GEMINI_API_KEY=...", code))
story.append(Paragraph(
    "Provider choice is automatic (generation/providers.get_llm_provider): Anthropic, then Gemini, then "
    "the mock. No code changes needed to switch.", body))

story.append(PageBreak())

# --- 3. Evaluation metrics ---
story.append(Paragraph("3. Evaluation methodology &amp; metrics", h1))
story.append(Paragraph("3.1 Five baselines (retrieval-only ablation)", h2))
baseline_rows = [
    ["#", "Baseline", "What it isolates"],
    ["1", "Fixed-size chunking + dense-only", "The naive \u201cchunk-and-embed\u201d approach"],
    ["2", "Structure-aware chunking + dense-only", "Value of structure-aware chunking alone"],
    ["3", "Hybrid (dense+BM25), no hierarchy", "Value of adding lexical/keyword retrieval"],
    ["4", "Hybrid + hierarchical retrieval", "Value of using the document tree"],
    ["5", "Hybrid + hierarchical + reranking", "Value of the reranking stage — the full pipeline"],
]
t = Table(baseline_rows, colWidths=[0.35 * inch, 2.3 * inch, 3.1 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2b4c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
]))
story.append(t)
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Reported per baseline: Precision@5, Hit@5 (a \u201crecall@5\u201d in the single-supporting-passage "
    "sense — did at least one relevant chunk make the top 5), MRR, NDCG@5, mean latency, and mean "
    "tokens retrieved.", body))

story.append(Paragraph("3.2 RAG-level metrics (full pipeline, by question category)", h2))
metric_rows = [
    ["Metric", "What it measures"],
    ["Context precision", "Of the chunks assembled into context, what fraction are actually relevant"],
    ["Context recall", "Did the assembled context include at least one supporting passage"],
    ["Answer relevance", "Vocabulary overlap between the answer and the question (on-topic check)"],
    ["Faithfulness", "Groundedness check passed — no fabricated citations or figures"],
    ["Answer correctness", "Fraction of expected keywords that appear in the generated answer"],
    ["Answerability accuracy", "Did the system correctly recognize answerable vs. unanswerable questions"],
]
t2 = Table(metric_rows, colWidths=[1.6 * inch, 4.15 * inch])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2b4c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
]))
story.append(t2)

story.append(Paragraph("3.3 A real, measured result (synthetic sample contract)", h2))
res_rows = [
    ["Baseline", "Hit@5", "Tokens retrieved"],
    ["1. Fixed-size chunking, dense-only", "0.87", "1106"],
    ["2. Structure-aware chunking, dense-only", "1.00", "237"],
]
t3 = Table(res_rows, colWidths=[3.2 * inch, 0.8 * inch, 1.7 * inch])
t3.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2b4c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
]))
story.append(t3)
story.append(Spacer(1, 6))
story.append(Paragraph(
    "Structure-aware chunking retrieves the right passage more reliably while using ~4.7x fewer tokens "
    "— the core architectural thesis, measured rather than asserted (rerun tests/test_evaluation.py to "
    "reproduce).", body))

story.append(PageBreak())

# --- 4. Real-world stress test ---
story.append(Paragraph("4. Real-world stress test", h1))
story.append(Paragraph(
    "Tested against two real documents: a 186-page UK government \u201cShort Form Contract\u201d "
    "(converted from .odt via LibreOffice headless) and a 117-page Everus Construction Group 2025 "
    "10-K annual report. Findings, not smoothed over:", body))
findings = [
    "Structure detection degrades on documents that don\u2019t use clean decimal numbering. The UK "
    "contract mixes Roman-numeral top sections with decimal sub-clauses; the 10-K uses \u201cItem 1\u201d / "
    "\u201cPART I\u201d conventions the heading detector has no pattern for.",
    "Two real bugs were caught and fixed during this test: bold financial-table column headers "
    "(\u201c2025 2024\u201d) and cover-page pull-quote figures (\u201c$3.75B\u201d) were being misclassified "
    "as section headings before a guard was added.",
    "On the UK contract, hierarchical retrieval measurably hurt Hit@5 (0.78 vs. 0.89) \u2014 direct "
    "evidence that hierarchy only helps when the tree is a decent representation of the real structure.",
    "On the 10-K, hybrid (dense+BM25) retrieval clearly helped over dense-only (Hit@5 0.75 vs. 0.62).",
    "Cross-reference extraction found almost nothing in the UK contract: its hyperlinked "
    "cross-references show the target clause\u2019s title as the link text (\u201cclause Data Protection "
    "and Security\u201d), not a section number \u2014 a different convention than the pattern this system "
    "looks for.",
    "pdfplumber\u2019s default table detector over-fires on text-heavy PDFs without visible gridlines "
    "(75 \u201ctables\u201d detected in the 10-K, many of which are actually justified paragraph text).",
]
story.append(ListFlowable([ListItem(Paragraph(f, body), leftIndent=10) for f in findings], bulletType="bullet"))

# --- 5. LLM provider guidance ---
story.append(Paragraph("5. Choosing an LLM provider: speed vs. accuracy vs. cost", h1))
story.append(Paragraph(
    "<b>Local model (Llama, Mistral, Qwen):</b> no per-query cost, full data privacy, no external rate "
    "limits — but CPU-only inference is not fast (several seconds to tens of seconds even for a short "
    "answer). A GPU gets a quantized 7-8B model to ~1-3 seconds, but that means provisioning and "
    "maintaining a GPU inference server \u2014 real extra ops for a project whose actual contribution is "
    "the retrieval architecture, not the generation model.", body))
story.append(Paragraph(
    "<b>Hosted fast-tier API model (Claude Haiku, Gemini Flash):</b> consistently 1-3 seconds, zero "
    "infra, strong quality even at the cheap tier \u2014 because in RAG, retrieval quality does most of "
    "the work. <b>Recommendation: a hosted fast-tier model</b>, specifically because the local route "
    "only reaches \u201cseconds\u201d with a GPU this project doesn\u2019t otherwise need.", body))
story.append(Paragraph(
    "<b>On free-tier Gemini rate limits:</b> real, not hypothetical \u2014 evaluate.py fires one call per "
    "question back-to-back and will hit low free-tier quotas. Both AnthropicProvider and GeminiProvider "
    "(generation/providers.py) now retry on HTTP 429/503 with exponential backoff, but for a timed demo "
    "specifically, a small pay-as-you-go credit on Claude Haiku is more reliable than a free tier. Verify "
    "current published rate limits on the provider\u2019s own docs before a demo \u2014 they change often.", body))
story.append(Paragraph(
    "Neither provider has been live-tested in the development sandbox (no Anthropic key was available; "
    "generativelanguage.googleapis.com is outright blocked by that sandbox\u2019s network egress rules). "
    "Test both with real keys in your own environment before relying on either.", small))

# --- 6. Known limitations ---
story.append(Paragraph("6. Known limitations (see README.md for the complete list)", h1))
limitations = [
    "Dense retrieval uses TF-IDF+SVD, not a transformer embedding model (disk constraints in the dev "
    "sandbox); a real embedding model is a one-line swap in vector_store/embedder.py.",
    "Reranking is feature-based (phrase overlap, term recall/precision), not a trained cross-encoder, "
    "for the same reason.",
    "Groundedness checking is lexical-overlap-based, not an NLI/entailment model.",
    "The pre-generation relevance threshold is heuristic and has an observed failure mode: a short "
    "rephrasing of an unanswerable question can slip past it (quantified by the answerability_accuracy "
    "metric, not hidden).",
    "Evaluation ground truth (expected_keyword_groups) is written by hand per document.",
]
story.append(ListFlowable([ListItem(Paragraph(f, body), leftIndent=10) for f in limitations], bulletType="bullet"))

story.append(Paragraph("7. Not yet built", h1))
story.append(Paragraph(
    "Phase 7: input-sanitization hardening beyond the prompt-level defense already in place, a FastAPI "
    "backend. Phase 8: a web UI (Streamlit/React). Everything above already works end to end from the "
    "command line without either.", body))

doc = SimpleDocTemplate("/home/claude/legal-rag-documentation.pdf", pagesize=letter,
                         topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                         leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                         title="Hierarchical Hybrid RAG — Project Documentation")
doc.build(story)
print("done")
