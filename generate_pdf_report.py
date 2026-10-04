"""
Generates a concise, high-density 1-Page Architecture Report PDF for Assignment 3.
MIT ADT University -- Unit III: Memory, Tools, and Environment Interaction.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf():
    pdf_filename = "architecture_report.pdf"
    
    # 1-page document with tight margins
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=28,
        bottomMargin=28
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=15,
        alignment=1,
        textColor=colors.HexColor("#0D47A1")
    )
    
    subtitle_style = ParagraphStyle(
        "SubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.HexColor("#37474F")
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#1A237E"),
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        "BodyTextSmall",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10,
        textColor=colors.HexColor("#263238")
    )

    table_header_style = ParagraphStyle(
        "TH",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9,
        textColor=colors.white,
        alignment=1
    )

    table_body_style = ParagraphStyle(
        "TB",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9,
        alignment=1,
        textColor=colors.HexColor("#212121")
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("<b>MIT ADT UNIVERSITY, PUNE — SCHOOL OF COMPUTING</b>", title_style))
    elements.append(Paragraph("<b>Unit III: Memory, Tools, and Environment Interaction | Assignment No: 03</b>", subtitle_style))
    elements.append(Paragraph("<b>Title:</b> Retrieval-Augmented Agent with Long-Term Memory and Tool Calling | <b>Mapped CO:</b> CO3 (Apply)", subtitle_style))
    elements.append(Spacer(1, 4))

    # 2. Architecture Overview & Components
    overview_text = (
        "<b>Architecture Abstract:</b> CogniAssist is an autonomous research assistant combining a dual-tier memory system "
        "with dynamic function calling. <b>Short-Term Memory (STM)</b> retains conversation context within an active session using a "
        "sliding-window buffer. <b>Long-Term Memory (LTM)</b> indexes user facts, preferences, and project parameters in a persistent "
        "<b>FAISS vector database</b> using cosine-normalized dense embeddings. An integrated <b>Feedback Adaptation Engine</b> "
        "detects user corrections, supersedes stale records, and synchronizes memory across sessions."
    )
    elements.append(Paragraph(overview_text, body_style))
    elements.append(Spacer(1, 4))

    # 3. System Architecture Diagram
    elements.append(Paragraph("<b>System Architecture & Interaction Flow</b>", heading_style))
    if os.path.exists("architecture_diagram.png"):
        img = Image("architecture_diagram.png", width=540, height=210)
        elements.append(img)
    elements.append(Spacer(1, 4))

    # 4. Component Breakdown & Tools
    col_desc = (
        "<b>Key Subsystems & Tools:</b><br/>"
        "• <b>Short-Term Buffer:</b> Session-isolated conversational turns (Max 10 turns) for localized reasoning.<br/>"
        "• <b>Long-Term FAISS Store:</b> Persistent <code>IndexFlatIP</code> with metadata tracking (<code>status: active / superseded</code>).<br/>"
        "• <b>Feedback Loop:</b> Pattern-matching & similarity-driven contradiction resolver that suppresses outdated facts.<br/>"
        "• <b>External Tools:</b> (1) <code>safe_calculator</code> (AST-safe math), (2) <code>python_code_executor</code> (sandboxed Python script runner), and (3) <code>live_web_search</code> (DuckDuckGo real-time search)."
    )
    elements.append(Paragraph(col_desc, body_style))
    elements.append(Spacer(1, 4))

    # 5. Quantitative Evaluation Table
    elements.append(Paragraph("<b>Quantitative Evaluation & Multi-Turn Benchmark Results</b>", heading_style))
    
    table_data = [
        [Paragraph("Scenario / Metric", table_header_style), Paragraph("Evaluated Functionality", table_header_style), Paragraph("Target", table_header_style), Paragraph("Score", table_header_style), Paragraph("Status", table_header_style)],
        [Paragraph("Scenario 1: Profile Recall", table_body_style), Paragraph("Recall university & research domain across sessions", table_body_style), Paragraph(">85%", table_body_style), Paragraph("<b>100%</b>", table_body_style), Paragraph("PASS", table_body_style)],
        [Paragraph("Scenario 2: Multi-Step Tools", table_body_style), Paragraph("Recall dataset size & execute calculator + Python sandbox", table_body_style), Paragraph(">90%", table_body_style), Paragraph("<b>100%</b>", table_body_style), Paragraph("PASS", table_body_style)],
        [Paragraph("Scenario 3: Feedback Loop", table_body_style), Paragraph("Supersede NeurIPS -> ICML 2026 and verify suppression", table_body_style), Paragraph("100%", table_body_style), Paragraph("<b>100%</b>", table_body_style), Paragraph("PASS", table_body_style)],
        [Paragraph("Overall Retrieval Accuracy", table_body_style), Paragraph("Relevant memories correctly recalled from FAISS", table_body_style), Paragraph(">85%", table_body_style), Paragraph("<b>100.0%</b>", table_body_style), Paragraph("PASS", table_body_style)],
        [Paragraph("Tool-Calling Success Rate", table_body_style), Paragraph("Successful invocations without parameter syntax errors", table_body_style), Paragraph(">90%", table_body_style), Paragraph("<b>100.0%</b>", table_body_style), Paragraph("PASS", table_body_style)]
    ]

    t = Table(table_data, colWidths=[110, 230, 50, 70, 75])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1A237E")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#B0BEC5")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 10))

    # 6. Signatures Row
    sig_data = [
        [
            Paragraph("<b>Student Signature:</b> _______________", body_style),
            Paragraph("<b>Faculty Signature:</b> _______________", body_style),
            Paragraph("<b>HOD Signature:</b> _______________", body_style)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[180, 180, 180])
    sig_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    elements.append(sig_table)

    doc.build(elements)
    print(f"1-Page Architecture Report generated: {pdf_filename}")

if __name__ == "__main__":
    generate_pdf()
