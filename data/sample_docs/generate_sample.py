"""
Generates a synthetic multi-page Master Service Agreement PDF with:
- nested numbered sections/subsections
- cross-references between sections
- a table (fee schedule)
- running header/footer with page numbers
This is test fixture data only - used to validate the ingestion pipeline.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib import colors

styles = getSampleStyleSheet()
h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13, spaceBefore=14, spaceAfter=6)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11.5, spaceBefore=10, spaceAfter=4)
h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=10.5, spaceBefore=8, spaceAfter=4)
body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6)

DOC_TITLE = "MASTER SERVICE AGREEMENT"

def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.75 * inch, 10.6 * inch, DOC_TITLE)
    canvas.drawRightString(7.75 * inch, 10.6 * inch, "CONFIDENTIAL")
    canvas.drawCentredString(4.25 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()

story = []
story.append(Paragraph(DOC_TITLE, styles["Title"]))
story.append(Paragraph("Effective Date: January 1, 2026", body))
story.append(Spacer(1, 12))

story.append(Paragraph("1. DEFINITIONS", h1))
story.append(Paragraph(
    "1.1 &nbsp;&nbsp;\"Agreement\" means this Master Service Agreement, including all "
    "exhibits, schedules, and amendments hereto.", body))
story.append(Paragraph(
    "1.2 &nbsp;&nbsp;\"Confidential Information\" means any non-public information disclosed "
    "by either Party, whether orally or in writing, that is designated as confidential.", body))
story.append(Paragraph(
    "1.3 &nbsp;&nbsp;\"Services\" means the professional services described in Section 2.1 "
    "and any applicable Statement of Work.", body))

story.append(Paragraph("2. SCOPE OF SERVICES", h1))
story.append(Paragraph("2.1 &nbsp;&nbsp;Description of Services", h2))
story.append(Paragraph(
    "The Vendor shall provide the Services described in each Statement of Work executed "
    "under this Agreement. Vendor shall perform the Services in a professional and "
    "workmanlike manner consistent with industry standards.", body))
story.append(Paragraph("2.2 &nbsp;&nbsp;Change Orders", h2))
story.append(Paragraph(
    "Any material change to the scope of Services must be documented in a written change "
    "order signed by both Parties, subject to the fee adjustments in Section 4.3.", body))

story.append(Paragraph("3. FEES AND PAYMENT", h1))
story.append(Paragraph("3.1 &nbsp;&nbsp;Payment Obligations", h2))
story.append(Paragraph(
    "Client shall pay all invoiced amounts within thirty (30) days of the invoice date. "
    "Late payments shall accrue interest at 1.5% per month, subject to Section 3.4.", body))
story.append(Paragraph("3.2 &nbsp;&nbsp;Fee Schedule", h2))
story.append(Paragraph("The applicable fees for each service tier are set out below:", body))

table_data = [
    ["Service Tier", "Monthly Fee (USD)", "Response SLA", "Included Hours"],
    ["Standard", "$4,500", "48 hours", "20"],
    ["Professional", "$9,000", "24 hours", "45"],
    ["Enterprise", "$18,500", "4 hours", "100"],
]
tbl = Table(table_data, colWidths=[1.6 * inch, 1.5 * inch, 1.3 * inch, 1.3 * inch])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
]))
story.append(tbl)
story.append(Spacer(1, 10))

story.append(Paragraph("3.3 &nbsp;&nbsp;Taxes", h2))
story.append(Paragraph(
    "All fees are exclusive of applicable taxes, which shall be borne by Client except "
    "taxes based on Vendor's net income.", body))
story.append(Paragraph("3.4 &nbsp;&nbsp;Disputed Invoices", h2))
story.append(Paragraph(
    "Client may withhold payment of a disputed amount only if it provides written notice "
    "of the dispute within fifteen (15) days of the invoice date, as further described in "
    "Section 6.2 (Notice Requirements).", body))

story.append(PageBreak())

story.append(Paragraph("4. TERM AND TERMINATION", h1))
story.append(Paragraph("4.1 &nbsp;&nbsp;Term", h2))
story.append(Paragraph(
    "This Agreement commences on the Effective Date and continues for an initial term of "
    "twenty-four (24) months, unless earlier terminated as set forth in this Section 4.", body))

story.append(Paragraph("4.2 &nbsp;&nbsp;Termination for Cause", h2))
story.append(Paragraph(
    "Either Party may terminate this Agreement upon written notice if the other Party "
    "materially breaches this Agreement and fails to cure such breach within thirty (30) "
    "days of receiving notice, subject to the notice requirements in Section 6.2. "
    "Notwithstanding the foregoing, Client may terminate immediately for Vendor's failure "
    "to maintain the insurance required under Section 8.1.", body))

story.append(Paragraph("4.2.1 &nbsp;&nbsp;Cure Period Exceptions", h3))
story.append(Paragraph(
    "No cure period shall apply to breaches of the confidentiality obligations in Section 7 "
    "or to a Party's insolvency, which shall each constitute grounds for immediate "
    "termination.", body))

story.append(Paragraph("4.3 &nbsp;&nbsp;Termination for Convenience", h2))
story.append(Paragraph(
    "Client may terminate this Agreement for convenience upon sixty (60) days' prior "
    "written notice. In the event of termination for convenience, Client shall pay Vendor "
    "for all Services performed through the effective date of termination, together with "
    "any non-cancellable costs reasonably incurred, but shall not be liable for lost "
    "profits.", body))

story.append(Paragraph("4.4 &nbsp;&nbsp;Effect of Termination", h2))
story.append(Paragraph(
    "Upon termination for any reason: (a) Client shall pay all outstanding fees for "
    "Services rendered prior to termination; (b) each Party shall return or destroy the "
    "other Party's Confidential Information; and (c) Sections 3, 5, 7, and 9 shall survive "
    "termination.", body))

story.append(Paragraph("5. LIMITATION OF LIABILITY", h1))
story.append(Paragraph(
    "5.1 &nbsp;&nbsp;EXCEPT FOR BREACHES OF SECTION 7 (CONFIDENTIALITY), NEITHER PARTY "
    "SHALL BE LIABLE FOR INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES. EACH PARTY'S "
    "TOTAL LIABILITY SHALL NOT EXCEED THE FEES PAID IN THE TWELVE (12) MONTHS PRECEDING "
    "THE CLAIM.", body))

story.append(Paragraph("6. NOTICES", h1))
story.append(Paragraph("6.1 &nbsp;&nbsp;Method of Notice", h2))
story.append(Paragraph(
    "All notices under this Agreement shall be in writing and delivered by email with "
    "confirmation of receipt, or by certified mail.", body))
story.append(Paragraph("6.2 &nbsp;&nbsp;Notice Requirements", h2))
story.append(Paragraph(
    "Notices relating to termination or disputed invoices must reference the specific "
    "section of this Agreement giving rise to the notice and must be sent to the address "
    "designated in the applicable Statement of Work.", body))

story.append(PageBreak())

story.append(Paragraph("7. CONFIDENTIALITY", h1))
story.append(Paragraph(
    "7.1 &nbsp;&nbsp;Each Party agrees to protect the other Party's Confidential "
    "Information using at least the same degree of care it uses to protect its own "
    "confidential information, and in no event less than reasonable care.", body))
story.append(Paragraph(
    "7.2 &nbsp;&nbsp;The obligations of this Section 7 shall survive termination of this "
    "Agreement for a period of five (5) years, as referenced in Section 4.4(c).", body))

story.append(Paragraph("8. INSURANCE AND INDEMNIFICATION", h1))
story.append(Paragraph("8.1 &nbsp;&nbsp;Insurance", h2))
story.append(Paragraph(
    "Vendor shall maintain commercial general liability insurance with limits of not less "
    "than $2,000,000 per occurrence throughout the Term.", body))
story.append(Paragraph("8.2 &nbsp;&nbsp;Indemnification", h2))
story.append(Paragraph(
    "Vendor shall indemnify Client against third-party claims arising from Vendor's gross "
    "negligence or willful misconduct in performing the Services, subject to the "
    "limitation of liability in Section 5.1.", body))

story.append(Paragraph("9. GENERAL PROVISIONS", h1))
story.append(Paragraph("9.1 &nbsp;&nbsp;Governing Law", h2))
story.append(Paragraph(
    "This Agreement is governed by the laws of the State of Delaware, without regard to "
    "its conflict of laws principles.", body))
story.append(Paragraph("9.2 &nbsp;&nbsp;Entire Agreement", h2))
story.append(Paragraph(
    "This Agreement, together with all Statements of Work, constitutes the entire "
    "agreement between the Parties and supersedes all prior agreements.", body))

doc = SimpleDocTemplate(
    "/home/claude/legal-rag/data/sample_docs/master_service_agreement.pdf",
    pagesize=letter,
    topMargin=0.9 * inch, bottomMargin=0.8 * inch,
    leftMargin=0.75 * inch, rightMargin=0.75 * inch,
)
doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print("Generated sample PDF.")
