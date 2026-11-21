#!/usr/bin/env python3
"""
Generate a 4-Slide Presentation for DTU B.Tech Project Mid-Term Evaluation
========================================================================
Project: Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code
Authors: Mankabir Singh (23/IT/199) & Mohnish Gautam (23/IT/200)
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation(output_path: str = "presentation.pptx"):
    prs = Presentation()
    # 16:9 Widescreen standard
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]  # Blank slide

    # Color Palette (Dark Premium Theme)
    BG_COLOR = RGBColor(15, 23, 42)        # Slate 900 #0f172a
    CARD_BG = RGBColor(30, 41, 59)        # Slate 800 #1e293b
    CARD_BORDER = RGBColor(51, 65, 85)    # Slate 700 #334155
    CYAN = RGBColor(56, 189, 248)         # Sky 400   #38bdf8
    PURPLE = RGBColor(192, 132, 252)      # Purple 400#c084fc
    GREEN = RGBColor(74, 222, 128)        # Green 400 #4ade80
    AMBER = RGBColor(251, 191, 36)        # Amber 400 #fbbf24
    TEXT_LIGHT = RGBColor(248, 250, 252)  # Slate 50  #f8fafc
    TEXT_MUTED = RGBColor(148, 163, 184)  # Slate 400 #94a3b8

    def add_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category_text="DTU B.TECH PROJECT-I EVALUATION"):
        # Category Badge / Header
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_c = cat_box.text_frame
        tf_c.word_wrap = True
        p_c = tf_c.paragraphs[0]
        p_c.text = category_text.upper()
        p_c.font.size = Pt(11)
        p_c.font.bold = True
        p_c.font.color.rgb = CYAN

        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
        tf_t = title_box.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(26)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_LIGHT

    # ==========================================
    # SLIDE 1: TITLE & PROJECT IDENTITY
    # ==========================================
    slide1 = prs.slides.add_slide(blank_layout)
    add_slide_background(slide1)

    # Top Pill Badge
    pill = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.8), Inches(4.2), Inches(0.4))
    pill.fill.solid()
    pill.fill.fore_color.rgb = CARD_BG
    pill.line.color.rgb = CYAN
    pill.line.width = Pt(1)
    p_pill = pill.text_frame.paragraphs[0]
    p_pill.text = "DELHI TECHNOLOGICAL UNIVERSITY • B.TECH PROJECT"
    p_pill.alignment = PP_ALIGN.CENTER
    p_pill.font.size = Pt(10)
    p_pill.font.bold = True
    p_pill.font.color.rgb = CYAN

    # Main Project Title
    title_box = slide1.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(1.8))
    tf1 = title_box.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Provenance-Aware Security Risk Assessment\nof Human–AI Hybrid Code"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_LIGHT

    sub_p = tf1.add_paragraph()
    sub_p.text = "Department of Information Technology • Mid-Term Progress Presentation"
    sub_p.font.size = Pt(16)
    sub_p.font.color.rgb = TEXT_MUTED

    # Info Cards Container (3 Cards side by side)
    card_w = Inches(3.64)
    card_h = Inches(2.6)
    top_y = Inches(3.8)

    # Card 1: Team Details
    c1 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), top_y, card_w, card_h)
    c1.fill.solid()
    c1.fill.fore_color.rgb = CARD_BG
    c1.line.color.rgb = CARD_BORDER
    tf_c1 = c1.text_frame
    tf_c1.word_wrap = True
    p = tf_c1.paragraphs[0]
    p.text = "👥 Team Members"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = CYAN

    p_m1 = tf_c1.add_paragraph()
    p_m1.text = "• Mankabir Singh\n  Roll No: 23/IT/199"
    p_m1.font.size = Pt(13)
    p_m1.font.color.rgb = TEXT_LIGHT

    p_m2 = tf_c1.add_paragraph()
    p_m2.text = "• Mohnish Gautam\n  Roll No: 23/IT/200"
    p_m2.font.size = Pt(13)
    p_m2.font.color.rgb = TEXT_LIGHT

    # Card 2: Group & Department
    c2 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.84), top_y, card_w, card_h)
    c2.fill.solid()
    c2.fill.fore_color.rgb = CARD_BG
    c2.line.color.rgb = CARD_BORDER
    tf_c2 = c2.text_frame
    tf_c2.word_wrap = True
    p = tf_c2.paragraphs[0]
    p.text = "🏷️ Group Information"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = PURPLE

    p_g1 = tf_c2.add_paragraph()
    p_g1.text = "• Group Number: [Insert Group No.]"
    p_g1.font.size = Pt(13)
    p_g1.font.color.rgb = TEXT_LIGHT

    p_g2 = tf_c2.add_paragraph()
    p_g2.text = "• Program: B.Tech (Information Technology)"
    p_g2.font.size = Pt(13)
    p_g2.font.color.rgb = TEXT_LIGHT

    p_g3 = tf_c2.add_paragraph()
    p_g3.text = "• Academic Year: 2026–2027"
    p_g3.font.size = Pt(13)
    p_g3.font.color.rgb = TEXT_MUTED

    # Card 3: Project Supervisor
    c3 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.88), top_y, card_w, card_h)
    c3.fill.solid()
    c3.fill.fore_color.rgb = CARD_BG
    c3.line.color.rgb = CARD_BORDER
    tf_c3 = c3.text_frame
    tf_c3.word_wrap = True
    p = tf_c3.paragraphs[0]
    p.text = "🎓 Project Supervisor"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = GREEN

    p_s1 = tf_c3.add_paragraph()
    p_s1.text = "• Supervisor:\n  [Prof. / Dr. Supervisor Name]"
    p_s1.font.size = Pt(13)
    p_s1.font.color.rgb = TEXT_LIGHT

    p_s2 = tf_c3.add_paragraph()
    p_s2.text = "• Department of Information Technology\n  Delhi Technological University (DTU)"
    p_s2.font.size = Pt(12)
    p_s2.font.color.rgb = TEXT_MUTED

    # ==========================================
    # SLIDE 2: PROGRESS SO FAR (CODE WRITTEN & EXECUTED)
    # ==========================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_slide_background(slide2)
    add_header(slide2, "Progress So Far — Code Written & Executed (Work Completed: 75%)")

    card2_w = Inches(5.66)
    card2_h = Inches(2.45)

    # Box 1: Static Security & AST Analysis
    b1 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), card2_w, card2_h)
    b1.fill.solid()
    b1.fill.fore_color.rgb = CARD_BG
    b1.line.color.rgb = CYAN
    tf_b1 = b1.text_frame
    tf_b1.word_wrap = True
    p = tf_b1.paragraphs[0]
    p.text = "1. Static Security Analysis Code (src/security/static_analysis.py)"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = CYAN

    p = tf_b1.add_paragraph()
    p.text = "• What we wrote: Built an AST rule scanner and integrated Flawfinder for C/C++ code."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_LIGHT

    p = tf_b1.add_paragraph()
    p.text = "• In simple terms: Scans code line-by-line to catch banned functions (strcpy, gets, sprintf)."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_LIGHT

    p = tf_b1.add_paragraph()
    p.text = "• Executed outcome: Flags exact line numbers, severity levels, and CWE vulnerability IDs."
    p.font.size = Pt(11)
    p.font.color.rgb = GREEN

    # Box 2: Model A (Code-Only ML Detector)
    b2 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.86), Inches(1.8), card2_w, card2_h)
    b2.fill.solid()
    b2.fill.fore_color.rgb = CARD_BG
    b2.line.color.rgb = PURPLE
    tf_b2 = b2.text_frame
    tf_b2.word_wrap = True
    p = tf_b2.paragraphs[0]
    p.text = "2. Model A Code (src/models/inference.py — Tri-Union SVM)"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = PURPLE

    p = tf_b2.add_paragraph()
    p.text = "• What we wrote: Trained an NLP machine learning classifier on 50,000 real-world C/C++ functions."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_LIGHT

    p = tf_b2.add_paragraph()
    p.text = "• In simple terms: Evaluates the source code text alone, strictly blind to who wrote it."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_LIGHT

    p = tf_b2.add_paragraph()
    p.text = "• Executed outcome: Achieved 79.1% validation accuracy and <20 ms fast CPU inference speed."
    p.font.size = Pt(11)
    p.font.color.rgb = GREEN

    # Box 3: Model B (Provenance-Aware Assessor)
    b3 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.55), card2_w, card2_h)
    b3.fill.solid()
    b3.fill.fore_color.rgb = CARD_BG
    b3.line.color.rgb = GREEN
    tf_b3 = b3.text_frame
    tf_b3.word_wrap = True
    p = tf_b3.paragraphs[0]
    p.text = "3. Model B Code (models/checkpoints/model_b_provenance_fusion.pkl)"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = GREEN

    p = tf_b3.add_paragraph()
    p.text = "• What we wrote: Mathematical fusion layer combining code risk with development history metadata."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_LIGHT

    p = tf_b3.add_paragraph()
    p.text = "• In simple terms: Uses AI involvement %, human edit cycles, and review status to adjust risk."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_LIGHT

    p = tf_b3.add_paragraph()
    p.text = "• Executed outcome: 96.9% accuracy on benchmark; slashed false alarms from 12 down to 0."
    p.font.size = Pt(11)
    p.font.color.rgb = GREEN

    # Box 4: Interactive Web Application
    b4 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.86), Inches(4.55), card2_w, card2_h)
    b4.fill.solid()
    b4.fill.fore_color.rgb = CARD_BG
    b4.line.color.rgb = AMBER
    tf_b4 = b4.text_frame
    tf_b4.word_wrap = True
    p = tf_b4.paragraphs[0]
    p.text = "4. Working Web Application (app.py — Streamlit Dashboard)"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = AMBER

    p = tf_b4.add_paragraph()
    p.text = "• What we wrote: Full web interface connecting Model A, Model B, and live code editors."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_LIGHT

    p = tf_b4.add_paragraph()
    p.text = "• In simple terms: Live testing tool where users test any custom code and see side-by-side results."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_LIGHT

    p = tf_b4.add_paragraph()
    p.text = "• Executed outcome: Tested & running live; auto-assigns Tier-1 Urgent vs Tier-3 Standard reviews."
    p.font.size = Pt(11)
    p.font.color.rgb = GREEN

    # ==========================================
    # SLIDE 3: TASKS LEFT TO BE DONE
    # ==========================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_slide_background(slide3)
    add_header(slide3, "Tasks Left to be Done (Next Steps in Simple Terms)")

    card3_w = Inches(3.64)
    card3_h = Inches(5.1)
    top3_y = Inches(1.8)

    # Task 1: Increase Model A Accuracy
    t1 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), top3_y, card3_w, card3_h)
    t1.fill.solid()
    t1.fill.fore_color.rgb = CARD_BG
    t1.line.color.rgb = CYAN
    tf_t1 = t1.text_frame
    tf_t1.word_wrap = True
    p = tf_t1.paragraphs[0]
    p.text = "🎯 Task 1\nIncrease Model A Accuracy on Modern C++"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = CYAN

    p = tf_t1.add_paragraph()
    p.text = "\n• Current Situation:\nModel A performs strongly on traditional C projects (85.6% on GPAC, 76.6% on Vim), but drops to 54.9% on Envoy Proxy (modern C++ with templates and virtual functions)."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_LIGHT

    p = tf_t1.add_paragraph()
    p.text = "\n• Task to do:\nFine-tune the code model on modern C++ syntax and object-oriented patterns."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_LIGHT

    p = tf_t1.add_paragraph()
    p.text = "\n• Expected Goal:\nPush Model A accuracy above 75% across all software projects, including modern C++."
    p.font.size = Pt(12)
    p.font.color.rgb = GREEN

    # Task 2: Train Model B on 1,000+ Dataset
    t2 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.84), top3_y, card3_w, card3_h)
    t2.fill.solid()
    t2.fill.fore_color.rgb = CARD_BG
    t2.line.color.rgb = PURPLE
    tf_t2 = t2.text_frame
    tf_t2.word_wrap = True
    p = tf_t2.paragraphs[0]
    p.text = "📊 Task 2\nTrain Model B on 1,000+ Sample Dataset"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = PURPLE

    p = tf_t2.add_paragraph()
    p.text = "\n• Current Situation:\nModel B has been verified on our locked benchmark, but the training corpus is currently limited to controlled seed tasks."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_LIGHT

    p = tf_t2.add_paragraph()
    p.text = "\n• Task to do:\nScale the controlled human-AI hybrid dataset to over 1,000+ verified code samples across 25+ vulnerability categories (integer overflows, memory leaks, null dereferences)."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_LIGHT

    p = tf_t2.add_paragraph()
    p.text = "\n• Expected Goal:\nRetrain Model B on this massive dataset to improve real-world generalization."
    p.font.size = Pt(12)
    p.font.color.rgb = GREEN

    # Task 3: Automated CI/CD Tooling
    t3 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.88), top3_y, card3_w, card3_h)
    t3.fill.solid()
    t3.fill.fore_color.rgb = CARD_BG
    t3.line.color.rgb = AMBER
    tf_t3 = t3.text_frame
    tf_t3.word_wrap = True
    p = tf_t3.paragraphs[0]
    p.text = "🤖 Task 3\nBuild Automated GitHub / CI/CD Bot"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = AMBER

    p = tf_t3.add_paragraph()
    p.text = "\n• Current Situation:\nModel B currently runs through our local Streamlit dashboard for demonstrations."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_LIGHT

    p = tf_t3.add_paragraph()
    p.text = "\n• Task to do:\nPackage the Model B risk assessor into a lightweight GitHub Actions bot that automatically scans new Pull Requests as developers commit code."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_LIGHT

    p = tf_t3.add_paragraph()
    p.text = "\n• Expected Goal:\nAutomatically tag pull requests with 'Tier-1 Urgent Audit' or 'Tier-3 Standard' in real-world workflows."
    p.font.size = Pt(12)
    p.font.color.rgb = GREEN

    # ==========================================
    # SLIDE 4: 1-MONTH COMPLETION TIMELINE
    # ==========================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_slide_background(slide4)
    add_header(slide4, "Expected Timeline of Project Completion (1-Month Plan)")

    card4_w = Inches(2.72)
    card4_h = Inches(5.1)
    top4_y = Inches(1.8)

    weeks_data = [
        ("Week 1", "Dataset Expansion to 1,000+ Samples", CYAN, [
            "• Expand human-AI hybrid code corpus to 1,000+ samples.",
            "• Cover 25+ CWE vulnerability types (buffer, memory, integer).",
            "• Verify strict 50/50 safe/vulnerable balance.",
            "• Check for zero data leakage."
        ]),
        ("Week 2", "Model A Modern C++ Adaptation", PURPLE, [
            "• Fine-tune Model A on modern C++ constructs (templates, classes).",
            "• Improve cross-project generalizability on Envoy Proxy.",
            "• Target accuracy >= 75% on unseen C++ codebases.",
            "• Preserve sub-20ms inference latency."
        ]),
        ("Week 3", "Model B Retraining & Evaluation", GREEN, [
            "• Retrain Model B fusion layer on the 1,000+ dataset.",
            "• Re-evaluate all ablation regimes (pure human vs raw AI).",
            "• Verify zero false positive rate on safe human code.",
            "• Benchmark triage ranking accuracy."
        ]),
        ("Week 4", "CI/CD Bot, Report & Final Viva", AMBER, [
            "• Build GitHub Actions bot for automated PR triage.",
            "• Complete final project report and documentation.",
            "• Conduct final system verification tests.",
            "• Ready for final B.Tech project presentation & defense."
        ]),
    ]

    for i, (w_title, w_sub, col, bullets) in enumerate(weeks_data):
        x = Inches(0.8 + i * 2.97)
        card = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, top4_y, card4_w, card4_h)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = col
        card.line.width = Pt(1.5)
        
        tf = card.text_frame
        tf.word_wrap = True
        
        # Week Header
        p_wh = tf.paragraphs[0]
        p_wh.text = f"📅 {w_title}"
        p_wh.font.bold = True
        p_wh.font.size = Pt(17)
        p_wh.font.color.rgb = col

        # Subtitle
        p_ws = tf.add_paragraph()
        p_ws.text = w_sub
        p_ws.font.bold = True
        p_ws.font.size = Pt(12)
        p_ws.font.color.rgb = TEXT_LIGHT

        # Bullets
        for b in bullets:
            p_b = tf.add_paragraph()
            p_b.text = b
            p_b.font.size = Pt(11)
            p_b.font.color.rgb = TEXT_MUTED

    prs.save(output_path)
    print(f"Presentation successfully created at: {output_path}")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "presentation.pptx"
    create_presentation(out)
