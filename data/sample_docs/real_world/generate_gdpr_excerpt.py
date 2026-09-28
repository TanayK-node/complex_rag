"""
Builds a PDF excerpt of Regulation (EU) 2016/679 (GDPR) — official, publicly
reusable EU legislative text (source: EUR-Lex / gdpr-info.eu) — covering
7 Articles across 4 Chapters. Chosen specifically because its numbering
convention (Chapter I / Article 4 / paragraph 1 / point (a)) is structurally
different from both documents already tested: not decimal ("4.2.1", the UK
contract) and not "Item 1A" (the 10-K). A genuine third structural pattern,
real operative legal text, not synthetic.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib import colors

styles = getSampleStyleSheet()
chapter_style = ParagraphStyle("Chapter", parent=styles["Heading1"], fontSize=13, spaceBefore=20, spaceAfter=4,
                                alignment=1, textColor=colors.HexColor("#1a2b4c"))
chapter_title_style = ParagraphStyle("ChapterTitle", parent=styles["Heading2"], fontSize=12, spaceBefore=2,
                                       spaceAfter=14, alignment=1)
article_style = ParagraphStyle("Article", parent=styles["Heading2"], fontSize=11.5, spaceBefore=14, spaceAfter=2)
article_title_style = ParagraphStyle("ArticleTitle", parent=styles["Heading3"], fontSize=10.5, spaceBefore=0,
                                       spaceAfter=8, textColor=colors.HexColor("#333333"))
body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, leading=14, spaceAfter=6)
point = ParagraphStyle("Point", parent=styles["Normal"], fontSize=9.5, leading=14, spaceAfter=5, leftIndent=18)
subpoint = ParagraphStyle("Subpoint", parent=styles["Normal"], fontSize=9.5, leading=14, spaceAfter=5, leftIndent=36)
note_style = ParagraphStyle("Note", parent=styles["Normal"], fontSize=8, leading=11, spaceAfter=4,
                             textColor=colors.HexColor("#666666"))

story = []
story.append(Paragraph("REGULATION (EU) 2016/679", chapter_style))
story.append(Paragraph(
    "General Data Protection Regulation — Excerpt (Articles 1, 4, 5, 6, 7, 17, 33) "
    "for RAG system testing purposes. Official text; source: EUR-Lex / Official Journal "
    "of the European Union, L 119/1, 4.5.2016.", note_style))
story.append(Spacer(1, 10))

def chapter(num, title):
    story.append(Paragraph(f"CHAPTER {num}", chapter_style))
    story.append(Paragraph(title, chapter_title_style))

def article(num, title):
    story.append(Paragraph(f"Article {num}", article_style))
    story.append(Paragraph(title, article_title_style))

def para(text):
    story.append(Paragraph(text, body))

def pt(num, text):
    story.append(Paragraph(f"{num}. {text}", point))

def subpt(letter_, text):
    story.append(Paragraph(f"({letter_}) {text}", subpoint))

# ---------------- CHAPTER I ----------------
chapter("I", "General provisions")

article("1", "Subject-matter and objectives")
pt(1, "This Regulation lays down rules relating to the protection of natural persons with regard to "
      "the processing of personal data and rules relating to the free movement of personal data.")
pt(2, "This Regulation protects fundamental rights and freedoms of natural persons and in particular "
      "their right to the protection of personal data.")
pt(3, "This Regulation applies to the processing of personal data by a controller not established in "
      "the Union, but in a place where Member State law applies by virtue of public international law.")

article("4", "Definitions")
para("For the purposes of this Regulation:")
pt(1, "\u2018personal data\u2019 means any information relating to an identified or identifiable natural "
      "person (\u2018data subject\u2019); an identifiable natural person is one who can be identified, "
      "directly or indirectly, in particular by reference to an identifier such as a name, an "
      "identification number, location data, an online identifier or to one or more factors specific "
      "to the physical, physiological, genetic, mental, economic, cultural or social identity of that "
      "natural person;")
pt(2, "\u2018processing\u2019 means any operation or set of operations which is performed on personal data "
      "or on sets of personal data, whether or not by automated means, such as collection, recording, "
      "organisation, structuring, storage, adaptation or alteration, retrieval, consultation, use, "
      "disclosure by transmission, dissemination or otherwise making available, alignment or "
      "combination, restriction, erasure or destruction;")
pt(7, "\u2018controller\u2019 means the natural or legal person, public authority, agency or other body "
      "which, alone or jointly with others, determines the purposes and means of the processing of "
      "personal data; where the purposes and means of such processing are determined by Union or "
      "Member State law, the controller or the specific criteria for its nomination may be provided "
      "for by Union or Member State law;")
pt(8, "\u2018processor\u2019 means a natural or legal person, public authority, agency or other body "
      "which processes personal data on behalf of the controller;")
pt(11, "\u2018consent\u2019 of the data subject means any freely given, specific, informed and unambiguous "
       "indication of the data subject's wishes by which he or she, by a statement or by a clear "
       "affirmative action, signifies agreement to the processing of personal data relating to him or her;")
pt(12, "\u2018personal data breach\u2019 means a breach of security leading to the accidental or unlawful "
       "destruction, loss, alteration, unauthorised disclosure of, or access to, personal data "
       "transmitted, stored or otherwise processed;")

story.append(PageBreak())

# ---------------- CHAPTER II ----------------
chapter("II", "Principles")

article("5", "Principles relating to processing of personal data")
pt(1, "Personal data shall be:")
subpt("a", "processed lawfully, fairly and in a transparent manner in relation to the data subject "
           "(\u2018lawfulness, fairness and transparency\u2019);")
subpt("b", "collected for specified, explicit and legitimate purposes and not further processed in a "
           "manner that is incompatible with those purposes (\u2018purpose limitation\u2019);")
subpt("c", "adequate, relevant and limited to what is necessary in relation to the purposes for which "
           "they are processed (\u2018data minimisation\u2019);")
subpt("d", "accurate and, where necessary, kept up to date; every reasonable step must be taken to "
           "ensure that personal data that are inaccurate are erased or rectified without delay "
           "(\u2018accuracy\u2019);")
subpt("e", "kept in a form which permits identification of data subjects for no longer than is "
           "necessary for the purposes for which the personal data are processed (\u2018storage "
           "limitation\u2019);")
subpt("f", "processed in a manner that ensures appropriate security of the personal data, including "
           "protection against unauthorised or unlawful processing and against accidental loss, "
           "destruction or damage, using appropriate technical or organisational measures "
           "(\u2018integrity and confidentiality\u2019).")
pt(2, "The controller shall be responsible for, and be able to demonstrate compliance with, "
      "paragraph 1 (\u2018accountability\u2019).")

article("6", "Lawfulness of processing")
pt(1, "Processing shall be lawful only if and to the extent that at least one of the following applies:")
subpt("a", "the data subject has given consent to the processing of his or her personal data for one "
           "or more specific purposes;")
subpt("b", "processing is necessary for the performance of a contract to which the data subject is "
           "party or in order to take steps at the request of the data subject prior to entering into "
           "a contract;")
subpt("c", "processing is necessary for compliance with a legal obligation to which the controller is "
           "subject;")
subpt("d", "processing is necessary in order to protect the vital interests of the data subject or of "
           "another natural person;")
subpt("e", "processing is necessary for the performance of a task carried out in the public interest "
           "or in the exercise of official authority vested in the controller;")
subpt("f", "processing is necessary for the purposes of the legitimate interests pursued by the "
           "controller or by a third party, except where such interests are overridden by the "
           "interests or fundamental rights and freedoms of the data subject which require protection "
           "of personal data, in particular where the data subject is a child.")

article("7", "Conditions for consent")
pt(1, "Where processing is based on consent, the controller shall be able to demonstrate that the "
      "data subject has consented to processing of his or her personal data.")
pt(2, "If the data subject's consent is given in the context of a written declaration which also "
      "concerns other matters, the request for consent shall be presented in a manner which is "
      "clearly distinguishable from the other matters, in an intelligible and easily accessible form, "
      "using clear and plain language.")
pt(3, "The data subject shall have the right to withdraw his or her consent at any time. The "
      "withdrawal of consent shall not affect the lawfulness of processing based on consent before "
      "its withdrawal. It shall be as easy to withdraw as to give consent.")

story.append(PageBreak())

# ---------------- CHAPTER III ----------------
chapter("III", "Rights of the data subject")

article("17", "Right to erasure (\u2018right to be forgotten\u2019)")
pt(1, "The data subject shall have the right to obtain from the controller the erasure of personal "
      "data concerning him or her without undue delay and the controller shall have the obligation to "
      "erase personal data without undue delay where one of the following grounds applies:")
subpt("a", "the personal data are no longer necessary in relation to the purposes for which they were "
           "collected or otherwise processed;")
subpt("b", "the data subject withdraws consent on which the processing is based and where there is no "
           "other legal ground for the processing;")
subpt("c", "the data subject objects to the processing and there are no overriding legitimate grounds "
           "for the processing;")
subpt("d", "the personal data have been unlawfully processed;")
subpt("e", "the personal data have to be erased for compliance with a legal obligation in Union or "
           "Member State law to which the controller is subject.")
pt(3, "Paragraphs 1 and 2 shall not apply to the extent that processing is necessary for exercising "
      "the right of freedom of expression and information, for compliance with a legal obligation, "
      "for reasons of public interest in the area of public health, for archiving purposes in the "
      "public interest, or for the establishment, exercise or defence of legal claims.")

story.append(PageBreak())

# ---------------- CHAPTER IV ----------------
chapter("IV", "Controller and processor")

article("33", "Notification of a personal data breach to the supervisory authority")
pt(1, "In the case of a personal data breach, the controller shall without undue delay and, where "
      "feasible, not later than 72 hours after having become aware of it, notify the personal data "
      "breach to the supervisory authority competent in accordance with Article 55, unless the "
      "personal data breach is unlikely to result in a risk to the rights and freedoms of natural "
      "persons. Where the notification to the supervisory authority is not made within 72 hours, it "
      "shall be accompanied by reasons for the delay.")
pt(2, "The processor shall notify the controller without undue delay after becoming aware of a "
      "personal data breach.")
pt(3, "The notification referred to in paragraph 1 shall at least:")
subpt("a", "describe the nature of the personal data breach including where possible, the categories "
           "and approximate number of data subjects concerned and the categories and approximate "
           "number of personal data records concerned;")
subpt("b", "communicate the name and contact details of the data protection officer or other contact "
           "point where more information can be obtained;")
subpt("c", "describe the likely consequences of the personal data breach;")
subpt("d", "describe the measures taken or proposed to be taken by the controller to address the "
           "personal data breach, including, where appropriate, measures to mitigate its possible "
           "adverse effects.")
pt(5, "The controller shall document any personal data breaches, comprising the facts relating to the "
      "personal data breach, its effects and the remedial action taken. That documentation shall "
      "enable the supervisory authority to verify compliance with this Article.")

doc = SimpleDocTemplate(
    "/home/claude/legal-rag/data/sample_docs/real_world/gdpr_excerpt.pdf",
    pagesize=letter, topMargin=0.8 * inch, bottomMargin=0.8 * inch,
    leftMargin=0.85 * inch, rightMargin=0.85 * inch,
)
doc.build(story)
print("done")
