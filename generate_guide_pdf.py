"""
Tạo file PDF hướng dẫn FraudShield cho người mới
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUT_PATH = "/mnt/user-data/outputs/FraudShield_HuongDan.pdf"
IMG_DIR  = "/home/claude/FraudShield/output"

W, H = A4

# ── Colors ────────────────────────────────────────────────────
C_DARK   = colors.HexColor("#0f1117")
C_NAVY   = colors.HexColor("#1e293b")
C_ACCENT = colors.HexColor("#4fc3f7")
C_TEXT   = colors.HexColor("#1e293b")
C_MUTED  = colors.HexColor("#64748b")
C_RED    = colors.HexColor("#dc2626")
C_GREEN  = colors.HexColor("#16a34a")
C_YELLOW = colors.HexColor("#d97706")
C_WHITE  = colors.white
C_LIGHT  = colors.HexColor("#f8fafc")
C_BORDER = colors.HexColor("#e2e8f0")
C_CODE   = colors.HexColor("#f1f5f9")

# ── Styles ────────────────────────────────────────────────────
def make_styles():
    s = {}
    base = dict(fontName="Helvetica", fontSize=11, leading=16,
                textColor=C_TEXT, spaceAfter=6)

    s["h1"] = ParagraphStyle("h1", fontName="Helvetica-Bold",
        fontSize=22, leading=28, textColor=C_DARK,
        spaceAfter=8, spaceBefore=20)

    s["h2"] = ParagraphStyle("h2", fontName="Helvetica-Bold",
        fontSize=15, leading=20, textColor=C_ACCENT,
        spaceAfter=6, spaceBefore=16,
        borderPad=4)

    s["h3"] = ParagraphStyle("h3", fontName="Helvetica-Bold",
        fontSize=12, leading=16, textColor=C_DARK,
        spaceAfter=4, spaceBefore=10)

    s["body"] = ParagraphStyle("body", fontName="Helvetica", fontSize=11,
        spaceAfter=6, textColor=C_TEXT, alignment=TA_JUSTIFY, leading=18)

    s["bullet"] = ParagraphStyle("bullet", fontName="Helvetica", fontSize=11,
        spaceAfter=6, textColor=C_TEXT, leftIndent=16, bulletIndent=4,
        leading=18)

    s["code"] = ParagraphStyle("code", fontName="Courier",
        fontSize=9.5, leading=14, textColor=C_DARK,
        backColor=C_CODE, leftIndent=12, rightIndent=12,
        spaceBefore=6, spaceAfter=6, borderPad=8,
        borderColor=C_BORDER, borderWidth=1, borderRadius=4)

    s["caption"] = ParagraphStyle("caption", fontName="Helvetica-Oblique",
        fontSize=9, leading=13, textColor=C_MUTED,
        alignment=TA_CENTER, spaceAfter=10)

    s["tag"] = ParagraphStyle("tag", fontName="Helvetica-Bold",
        fontSize=10, leading=14, textColor=C_WHITE,
        backColor=C_ACCENT, leftIndent=8, rightIndent=8,
        spaceBefore=2, spaceAfter=2, borderPad=4)

    s["cover_title"] = ParagraphStyle("cover_title",
        fontName="Helvetica-Bold", fontSize=32, leading=40,
        textColor=C_WHITE, alignment=TA_CENTER, spaceAfter=12)

    s["cover_sub"] = ParagraphStyle("cover_sub",
        fontName="Helvetica", fontSize=14, leading=20,
        textColor=colors.HexColor("#94a3b8"), alignment=TA_CENTER)

    s["cover_label"] = ParagraphStyle("cover_label",
        fontName="Helvetica-Bold", fontSize=11, leading=16,
        textColor=C_ACCENT, alignment=TA_CENTER, spaceBefore=4)

    s["step_num"] = ParagraphStyle("step_num",
        fontName="Helvetica-Bold", fontSize=16, leading=20,
        textColor=C_ACCENT, alignment=TA_CENTER)

    s["step_title"] = ParagraphStyle("step_title",
        fontName="Helvetica-Bold", fontSize=12, leading=16,
        textColor=C_DARK)

    s["step_body"] = ParagraphStyle("step_body",
        fontName="Helvetica", fontSize=10.5, leading=16,
        textColor=C_TEXT, alignment=TA_JUSTIFY)

    s["toc"] = ParagraphStyle("toc", fontName="Helvetica",
        fontSize=11, leading=20, textColor=C_TEXT, leftIndent=0)

    s["toc_indent"] = ParagraphStyle("toc_indent", fontName="Helvetica",
        fontSize=10, leading=18, textColor=C_MUTED, leftIndent=16)

    return s

S = make_styles()


def hr(color=C_BORDER, thickness=0.5):
    return HRFlowable(width="100%", thickness=thickness,
                      color=color, spaceAfter=8, spaceBefore=4)


def img(name, width=14*cm, caption=None):
    path = os.path.join(IMG_DIR, name)
    items = []
    if os.path.exists(path):
        items.append(Image(path, width=width, height=width*0.48))
        if caption:
            items.append(Paragraph(f"<i>{caption}</i>", S["caption"]))
    return items


def section_box(title, color=C_ACCENT):
    """Thanh tiêu đề section màu nổi bật."""
    return Table([[Paragraph(f"<b>{title}</b>",
                    ParagraphStyle("sh", fontName="Helvetica-Bold",
                        fontSize=13, textColor=C_WHITE, leading=18))]],
        colWidths=[17*cm],
        style=TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), color),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [color]),
            ("TOPPADDING", (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING", (0,0), (-1,-1), 14),
            ("RIGHTPADDING", (0,0), (-1,-1), 14),
            ("ROUNDEDCORNERS", [6]),
        ]))


def info_box(text, bg=C_LIGHT, border=C_ACCENT):
    return Table([[Paragraph(text, ParagraphStyle("ib",
                    fontName="Helvetica", fontSize=10.5, leading=16,
                    textColor=C_TEXT))]],
        colWidths=[17*cm],
        style=TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), bg),
            ("LINEAFTER", (0,0), (0,-1), 3, border),
            ("LINEBEFORE", (0,0), (0,-1), 3, border),
            ("TOPPADDING", (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("LEFTPADDING", (0,0), (-1,-1), 14),
            ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ]))


def step_card(num, title, body_lines, color=C_ACCENT):
    body_text = "<br/>".join(f"• {l}" for l in body_lines)
    data = [[
        Paragraph(str(num), ParagraphStyle("sn", fontName="Helvetica-Bold",
            fontSize=20, textColor=C_WHITE, alignment=TA_CENTER, leading=24)),
        [Paragraph(f"<b>{title}</b>", ParagraphStyle("st",
            fontName="Helvetica-Bold", fontSize=11.5, textColor=C_DARK, leading=16)),
         Spacer(1, 4),
         Paragraph(body_text, ParagraphStyle("sb",
            fontName="Helvetica", fontSize=10, textColor=C_TEXT,
            leading=15, alignment=TA_LEFT))]
    ]]
    return Table(data, colWidths=[1.4*cm, 15.6*cm],
        style=TableStyle([
            ("BACKGROUND", (0,0), (0,0), color),
            ("BACKGROUND", (1,0), (1,0), C_LIGHT),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("TOPPADDING", (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("LEFTPADDING", (0,0), (0,0), 4),
            ("RIGHTPADDING", (0,0), (0,0), 4),
            ("LEFTPADDING", (1,0), (1,0), 12),
            ("ROUNDEDCORNERS", [6]),
            ("BOX", (0,0), (-1,-1), 1, C_BORDER),
        ]))


def kpi_row(items):
    """items = list of (value, label, color)"""
    cells = [[
        [Paragraph(f"<b>{v}</b>", ParagraphStyle("kv",
            fontName="Helvetica-Bold", fontSize=18, textColor=c,
            alignment=TA_CENTER, leading=22)),
         Paragraph(lbl, ParagraphStyle("kl",
            fontName="Helvetica", fontSize=9, textColor=C_MUTED,
            alignment=TA_CENTER, leading=12))]
        for v, lbl, c in items
    ]]
    n = len(items)
    w = 17*cm / n
    return Table(cells, colWidths=[w]*n,
        style=TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), C_LIGHT),
            ("GRID", (0,0), (-1,-1), 1, C_BORDER),
            ("TOPPADDING", (0,0), (-1,-1), 12),
            ("BOTTOMPADDING", (0,0), (-1,-1), 12),
            ("ROUNDEDCORNERS", [6]),
        ]))


# ═══════════════════════════════════════════════════════════════
# BUILD DOCUMENT
# ═══════════════════════════════════════════════════════════════
def build():
    doc = SimpleDocTemplate(
        OUT_PATH, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    story = []

    # ─────────────────────────────────────────────────────────
    # TRANG BÌA
    # ─────────────────────────────────────────────────────────
    cover_bg = Table(
        [[Paragraph("FRAUDSHIELD", ParagraphStyle("ct",
            fontName="Helvetica-Bold", fontSize=36, textColor=C_WHITE,
            alignment=TA_CENTER, leading=42)),
          ],
         [Paragraph("He thong Phat hien Gian lan Ngan hang", ParagraphStyle("cs",
            fontName="Helvetica", fontSize=14, textColor=colors.HexColor("#94a3b8"),
            alignment=TA_CENTER, leading=20)),
          ],
         [Paragraph("Huong dan Toan dien cho Nguoi moi Bat dau", ParagraphStyle("cs2",
            fontName="Helvetica-Bold", fontSize=12, textColor=C_ACCENT,
            alignment=TA_CENTER, leading=18)),
          ],
        ],
        colWidths=[17*cm],
        style=TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), C_DARK),
            ("TOPPADDING", (0,0), (-1,-1), 18),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("LEFTPADDING", (0,0), (-1,-1), 20),
            ("RIGHTPADDING", (0,0), (-1,-1), 20),
            ("ROUNDEDCORNERS", [10]),
        ])
    )
    story.append(cover_bg)
    story.append(Spacer(1, 0.5*cm))

    # KPI tóm tắt
    story.append(kpi_row([
        ("50,000", "Giao dich phan tich", C_ACCENT),
        ("4.85%", "Fraud Rate", C_RED),
        ("3 tabs", "Dashboard truc quan", C_GREEN),
        ("Zero-touch", "Tu dong hoa", C_YELLOW),
    ]))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "FAA4023 — Data Analysis in Accounting  |  Final Assignment 2025–2026",
        ParagraphStyle("footer_cover", fontName="Helvetica", fontSize=9,
            textColor=C_MUTED, alignment=TA_CENTER)))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────
    # MUC LUC
    # ─────────────────────────────────────────────────────────
    story.append(section_box("MUC LUC"))
    story.append(Spacer(1, 0.3*cm))
    toc_items = [
        ("1", "TONG QUAN SAN PHAM", False),
        ("", "FraudShield la gi? Tai sao can?", True),
        ("", "Du lieu dau vao (50,000 giao dich)", True),
        ("2", "CO CHE HOAT DONG", False),
        ("", "Buoc 1 — Lam sach du lieu tu dong", True),
        ("", "Buoc 2 — Tao Features moi", True),
        ("", "Buoc 3 — Huan luyen mo hinh AI", True),
        ("", "Buoc 4 — Phan loai rui ro", True),
        ("3", "WEB DASHBOARD — 4 TABS", False),
        ("", "Tab 1: Tong quan Dataset", True),
        ("", "Tab 2: Phan tich Rui ro", True),
        ("", "Tab 3: Du doan Tu dong", True),
        ("", "Tab 4: Scheduler Monitor", True),
        ("4", "TU DONG HOA ZERO-TOUCH", False),
        ("", "Scheduler.py — scan & xu ly ngam", True),
        ("", "Email Alert Gmail tu dong", True),
        ("5", "KET QUA THUC TE", False),
        ("6", "HUONG DAN CAI DAT & CHAY", False),
    ]
    for num, title, indent in toc_items:
        style = S["toc_indent"] if indent else S["toc"]
        prefix = f"    {title}" if indent else f"<b>{num}. {title}</b>"
        story.append(Paragraph(prefix, style))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────
    # PHAN 1 — TONG QUAN
    # ─────────────────────────────────────────────────────────
    story.append(section_box("1. TONG QUAN SAN PHAM"))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("<b>FraudShield la gi?</b>", S["h3"]))
    story.append(Paragraph(
        "FraudShield la mot he thong phat hien gian lan ngan hang <b>hoan toan tu dong</b>. "
        "He thong su dung tri tue nhan tao (Machine Learning) de quet va phan tich tung "
        "giao dich, tu dong phan loai muc do rui ro, va gui canh bao email ngay khi phat hien "
        "giao dich bat thuong — <b>khong can nguoi dung lam bat cu thu gi</b> sau khi khoi dong.",
        S["body"]))
    story.append(Spacer(1, 0.2*cm))

    story.append(info_box(
        "<b>Vi du thuc te:</b> Ngan hang nhan duoc 10,000 giao dich moi ngay. "
        "Kiem tra tay la khong the. FraudShield tu dong xu ly toan bo trong vai phut, "
        "chi canh bao nhung giao dich thuc su dang nghi ngo — giam tai kho khan cho nhan vien.",
        bg=colors.HexColor("#eff6ff"), border=C_ACCENT))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("<b>Du lieu dau vao</b>", S["h3"]))
    data_table = Table([
        [Paragraph("<b>Thong tin</b>", ParagraphStyle("th", fontName="Helvetica-Bold",
            fontSize=10, textColor=C_WHITE)),
         Paragraph("<b>Chi tiet</b>", ParagraphStyle("th", fontName="Helvetica-Bold",
            fontSize=10, textColor=C_WHITE))],
        ["Nguon du lieu", "FraudShield_Banking_Data.csv (Kaggle)"],
        ["So luong", "50,000 giao dich (49,996 sau lam sach)"],
        ["So cot", "25 cot goc + 18 cot moi tao ra"],
        ["Khoang thoi gian", "Jan – May 2025"],
        ["Ty le Fraud", "4.85% (2,423 giao dich gian lan)"],
        ["Missing values", "150 o truong (tu dong xu ly)"],
    ], colWidths=[6*cm, 11*cm],
    style=TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C_DARK),
        ("TEXTCOLOR", (0,0), (-1,0), C_WHITE),
        ("BACKGROUND", (0,1), (-1,-1), C_WHITE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_WHITE, C_LIGHT]),
        ("GRID", (0,0), (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(data_table)
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────
    # PHAN 2 — CO CHE
    # ─────────────────────────────────────────────────────────
    story.append(section_box("2. CO CHE HOAT DONG — PIPELINE TU DONG"))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Toan bo quy trinh gom 4 buoc lien tiep, chay hoan toan tu dong khi co file CSV moi:",
        S["body"]))
    story.append(Spacer(1, 0.2*cm))
    story += img("chart4_pipeline.png", width=16*cm,
                 caption="Hinh 1 — Luong xu ly tu dong (Zero-touch Pipeline)")
    story.append(Spacer(1, 0.3*cm))

    # Buoc 1
    story.append(KeepTogether([
        step_card(1, "Lam sach du lieu — 01_data_cleaning.py", [
            "Doc file CSV goc (50,000 dong)",
            "Xoa 4 dong thieu nhan Fraud_Label (khong the doan duoc)",
            "Dien 146 o trong: so dung median, chu dung mode",
            "Tach Transaction_Date → Month, DayOfWeek",
            "Tach Transaction_Time → Hour (0–23)",
            "Chuan hoa Yes/No cho cac cot Is_International, Unusual_Time",
            "Ket qua: 49,996 dong sach — xuat cleaned_data.csv",
        ]),
        Spacer(1, 0.2*cm),
    ]))
    story.append(Paragraph("<b>Log tu dong in ra sau khi chay:</b>", S["h3"]))
    story.append(Paragraph(
        "[✓] Doc 50,000 dong tu: FraudShield_Banking_Data.csv<br/>"
        "[✓] Xoa 4 dong thieu nhan Fraud_Label<br/>"
        "[✓] Impute 146 gia tri thieu (so: median, phan loai: mode)<br/>"
        "[✓] Tach Transaction_Date → Month, DayOfWeek, DayName<br/>"
        "[✓] Tach Transaction_Time → Hour (0–23)<br/>"
        "[✓] Xuat 49,996 dong sach → cleaned_data.csv",
        S["code"]))
    story.append(Spacer(1, 0.3*cm))

    # Buoc 2
    story.append(KeepTogether([
        step_card(2, "Tao Features moi — 02_feature_engineering.py", [
            "is_high_risk_combo: quoc te + gio bat thuong → fraud rate 7.3%",
            "amount_vs_avg_ratio: giao dich dot bien so voi lich su",
            "balance_vs_amount_ratio: so du tai khoan / gia tri giao dich",
            "is_night: giao dich luc 22h–6h sang",
            "transaction_velocity: hom nay giao dich nhieu bat thuong?",
            "is_large_transaction: gia tri cao nhat trong 24h gan nhat",
            "Encode 12 cot phan loai thanh so de mo hinh hoc",
        ]),
        Spacer(1, 0.2*cm),
    ]))

    feat_table = Table([
        [Paragraph("<b>Feature</b>", ParagraphStyle("th2", fontName="Helvetica-Bold",
            fontSize=10, textColor=C_WHITE)),
         Paragraph("<b>Cong thuc</b>", ParagraphStyle("th2", fontName="Helvetica-Bold",
            fontSize=10, textColor=C_WHITE)),
         Paragraph("<b>Y nghia</b>", ParagraphStyle("th2", fontName="Helvetica-Bold",
            fontSize=10, textColor=C_WHITE))],
        ["is_high_risk_combo", "Intl=Yes AND UnusualTime=Yes", "To hop rui ro nhat (7.3%)"],
        ["amount_vs_avg_ratio", "Amount / Avg_Amount", "Giao dich dot bien"],
        ["balance_vs_amount_ratio", "Balance / Amount", "Giao dich lon so voi so du"],
        ["is_night", "Hour<6 OR Hour>22", "Giao dich dem khuya"],
        ["transaction_velocity", "Daily / Weekly * 7", "Tan suat giao dich bat thuong"],
        ["is_large_transaction", "Amount > Max_Last_24h", "Lon nhat trong 24 gio"],
    ], colWidths=[5*cm, 6*cm, 6*cm],
    style=TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C_DARK),
        ("TEXTCOLOR", (0,0), (-1,0), C_WHITE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_WHITE, C_LIGHT]),
        ("GRID", (0,0), (-1,-1), 0.5, C_BORDER),
        ("FONTNAME", (0,1), (0,-1), "Courier"),
        ("FONTSIZE", (0,0), (-1,-1), 9.5),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 0.3*cm))

    # Buoc 3
    story.append(KeepTogether([
        step_card(3, "Huan luyen mo hinh AI — 03_train_model.py", [
            "Thuat toan: Random Forest (rung ngau nhien 100 cay quyet dinh)",
            "Chia du lieu: 80% huan luyen / 20% kiem tra (stratified)",
            "Xu ly mat can bang: class_weight='balanced' (chi 4.85% la Fraud)",
            "Xuat mo hinh: fraud_model.pkl (dung cho moi lan du doan sau)",
            "Xuat bieu do: Confusion Matrix, ROC Curve, Feature Importance",
        ]),
        Spacer(1, 0.2*cm),
    ]))

    story.append(info_box(
        "<b>Tai sao dung Random Forest?</b><br/>"
        "• Khong can chuan hoa so lieu (nen chay nhanh)<br/>"
        "• Tu dong xu ly du lieu mat can bang (fraud it hon nhieu)<br/>"
        "• De giai thich: biet duoc cot nao quan trong nhat<br/>"
        "• Robuste: khong bi qua khop (overfitting) nhu Neural Network<br/>"
        "<br/>"
        "<b>Tai sao dung Recall lam chi tieu chinh?</b><br/>"
        "Trong gian lan ngan hang, <b>bo sot 1 vu Fraud</b> (tiet lo tien that) nguy hiem hon "
        "nhieu so voi <b>canh bao nham</b> 1 giao dich binh thuong (chi mat cong kiem tra them).",
        bg=colors.HexColor("#eff6ff"), border=C_ACCENT))
    story.append(Spacer(1, 0.3*cm))

    # Buoc 4
    story.append(KeepTogether([
        step_card(4, "Phan loai Rui ro & Bao cao", [
            "Moi giao dich duoc tinh Fraud_Probability: 0.0 → 1.0",
            "< 30%: LOW (xanh la) — giao dich binh thuong",
            "30–60%: MEDIUM (vang) — can theo doi",
            "> 60%: HIGH (do) — nghi ngo fraud, kiem tra ngay",
            "Xuat Excel 3 sheets: tat ca, chi HIGH, tong hop",
            "Gui Gmail alert tu dong neu co giao dich HIGH",
        ]),
        Spacer(1, 0.4*cm),
    ]))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────
    # PHAN 3 — DASHBOARD
    # ─────────────────────────────────────────────────────────
    story.append(section_box("3. WEB DASHBOARD — GIAO DIEN TRUC QUAN"))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Dashboard duoc xay dung bang Streamlit — mot thu vien Python cho phep tao "
        "web app chi bang code Python thuan tuy, khong can biet HTML hay JavaScript. "
        "Chay lenh <b>streamlit run app.py</b> se tu dong mo trinh duyet tai "
        "<b>localhost:8501</b>.",
        S["body"]))
    story.append(Spacer(1, 0.2*cm))

    story.append(info_box(
        "<b>Streamlit la gi?</b>  Tuong tu Google Colab nhung tao ra website that su "
        "co the chia se. Nguoi dung khong can cai gi ca — chi mo link tren trinh duyet.",
        bg=colors.HexColor("#f0fdf4"), border=C_GREEN))
    story.append(Spacer(1, 0.3*cm))

    tabs_data = [
        [Paragraph("<b>Tab</b>", ParagraphStyle("th3", fontName="Helvetica-Bold",
            fontSize=10, textColor=C_WHITE)),
         Paragraph("<b>Noi dung</b>", ParagraphStyle("th3", fontName="Helvetica-Bold",
            fontSize=10, textColor=C_WHITE)),
         Paragraph("<b>Bieu do chinh</b>", ParagraphStyle("th3", fontName="Helvetica-Bold",
            fontSize=10, textColor=C_WHITE))],
        ["Tab 1: Tong quan", "KPI cards, phan bo tong the",
         "Bar chart, Pie chart, Heatmap, Line chart"],
        ["Tab 2: Phan tich", "Loc du lieu theo nhieu tieu chi",
         "Scatter plot, Box plot, Data table"],
        ["Tab 3: Du doan", "Upload CSV → du doan → download Excel",
         "Risk table mau sac, Feature Importance"],
        ["Tab 4: Scheduler", "Xem trang thai tu dong, log realtime",
         "Log viewer, inbox status, upload vao inbox"],
    ]
    tabs_table = Table(tabs_data, colWidths=[4*cm, 6.5*cm, 6.5*cm],
    style=TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C_DARK),
        ("TEXTCOLOR", (0,0), (-1,0), C_WHITE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_WHITE, C_LIGHT]),
        ("GRID", (0,0), (-1,-1), 0.5, C_BORDER),
        ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("TOPPADDING", (0,0), (-1,-1), 9),
        ("BOTTOMPADDING", (0,0), (-1,-1), 9),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(tabs_table)
    story.append(Spacer(1, 0.4*cm))

    # Charts thuc te
    story.append(Paragraph("<b>Ket qua EDA thuc te tu du lieu</b>", S["h3"]))
    story += img("chart1_fraud_by_txtype.png", width=15*cm,
                 caption="Hinh 2 — Fraud Rate theo loai giao dich (ATM / Online / POS)")
    story += img("chart2_heatmap_combo.png", width=12*cm,
                 caption="Hinh 3 — Fraud Rate theo to hop Quoc te x Gio giao dich")
    story += img("chart3_monthly_trend.png", width=15*cm,
                 caption="Hinh 4 — Xu huong Fraud theo thang (Jan–May 2025)")
    story.append(PageBreak())

    # Model evaluation charts
    story.append(section_box("4. DANH GIA MO HINH AI", color=colors.HexColor("#1e3a5f")))
    story.append(Spacer(1, 0.3*cm))
    story += img("confusion_matrix_roc.png", width=16*cm,
                 caption="Hinh 5 — Confusion Matrix va ROC Curve cua mo hinh Random Forest")
    story.append(Spacer(1, 0.2*cm))
    story += img("feature_importance.png", width=15*cm,
                 caption="Hinh 6 — Top 10 Features quan trong nhat trong viec phat hien Fraud")
    story.append(Spacer(1, 0.3*cm))

    story.append(info_box(
        "<b>Doc hieu Confusion Matrix:</b><br/>"
        "• True Negative (trai tren): du doan Normal — thuc te Normal → DUNG<br/>"
        "• True Positive (phai duoi): du doan Fraud — thuc te Fraud → DUNG<br/>"
        "• False Positive: canh bao nham giao dich binh thuong (chap nhan duoc)<br/>"
        "• False Negative: <b>BO SOT giao dich Fraud</b> → day la sai lam nguy hiem nhat",
        bg=colors.HexColor("#fff7ed"), border=C_YELLOW))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────
    # PHAN 4 — TU DONG HOA
    # ─────────────────────────────────────────────────────────
    story.append(section_box("5. TU DONG HOA ZERO-TOUCH — scheduler.py"))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Day la diem manh nhat cua FraudShield: sau khi khoi dong, he thong tu dong lam "
        "moi thu <b>khong can nguoi dung can thiep</b>.", S["body"]))
    story.append(Spacer(1, 0.2*cm))

    auto_flow = Table([
        [Paragraph("Hanh dong cua Nguoi dung", ParagraphStyle("af1",
            fontName="Helvetica-Bold", fontSize=11, textColor=C_WHITE,
            alignment=TA_CENTER)),
         Paragraph("He thong tu dong xu ly", ParagraphStyle("af2",
            fontName="Helvetica-Bold", fontSize=11, textColor=C_WHITE,
            alignment=TA_CENTER))],
        [Paragraph("1. Chay lenh:<br/><b>python scheduler.py</b>",
                   ParagraphStyle("afc", fontName="Helvetica", fontSize=10,
                       textColor=C_TEXT, leading=16)),
         Paragraph("He thong khoi dong, bat dau scan thu muc inbox/ moi 60 giay",
                   ParagraphStyle("afc2", fontName="Helvetica", fontSize=10,
                       textColor=C_TEXT, leading=16))],
        [Paragraph("2. Tha file CSV vao thu muc <b>inbox/</b>",
                   ParagraphStyle("afc", fontName="Helvetica", fontSize=10,
                       textColor=C_TEXT, leading=16)),
         Paragraph("Scheduler phat hien file moi trong lan scan tiep theo",
                   ParagraphStyle("afc2", fontName="Helvetica", fontSize=10,
                       textColor=C_TEXT, leading=16))],
        [Paragraph("(Khong lam gi them)",
                   ParagraphStyle("afc", fontName="Helvetica-Oblique", fontSize=10,
                       textColor=C_MUTED, leading=16)),
         Paragraph("Tu dong: Lam sach → Features → Du doan → Phan loai risk",
                   ParagraphStyle("afc2", fontName="Helvetica", fontSize=10,
                       textColor=C_TEXT, leading=16))],
        [Paragraph("(Khong lam gi them)",
                   ParagraphStyle("afc", fontName="Helvetica-Oblique", fontSize=10,
                       textColor=C_MUTED, leading=16)),
         Paragraph("Tu dong: Xuat bao cao Excel vao output/",
                   ParagraphStyle("afc2", fontName="Helvetica", fontSize=10,
                       textColor=C_TEXT, leading=16))],
        [Paragraph("3. Kiem tra hop thu Gmail",
                   ParagraphStyle("afc", fontName="Helvetica-Bold", fontSize=10,
                       textColor=C_TEXT, leading=16)),
         Paragraph("Tu dong: Gui Gmail alert kem file Excel dinh kem",
                   ParagraphStyle("afc2", fontName="Helvetica-Bold", fontSize=10,
                       textColor=C_RED, leading=16))],
    ], colWidths=[6.5*cm, 10.5*cm],
    style=TableStyle([
        ("BACKGROUND", (0,0), (0,0), C_DARK),
        ("BACKGROUND", (1,0), (1,0), C_DARK),
        ("BACKGROUND", (0,1), (0,-1), colors.HexColor("#f0fdf4")),
        ("BACKGROUND", (1,1), (1,-1), colors.HexColor("#eff6ff")),
        ("LINEAFTER", (0,0), (0,-1), 1, C_BORDER),
        ("GRID", (0,0), (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(auto_flow)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("<b>Cau hinh email trong config.py:</b>", S["h3"]))
    story.append(Paragraph(
        "EMAIL_SENDER   = 'your_email@gmail.com'<br/>"
        "EMAIL_PASSWORD = 'xxxx xxxx xxxx xxxx'  # App Password 16 ky tu<br/>"
        "EMAIL_RECEIVER = 'receiver@gmail.com'<br/>"
        "SCAN_INTERVAL_SECONDS = 60  # Scan moi 60 giay",
        S["code"]))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────
    # PHAN 5 — CAI DAT
    # ─────────────────────────────────────────────────────────
    story.append(section_box("6. HUONG DAN CAI DAT & CHAY", color=C_GREEN))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph(
        "Yeu cau: Python 3.9+, VS Code (khuyen nghi), ket noi Internet.",
        S["body"]))
    story.append(Spacer(1, 0.2*cm))

    installs = [
        ("Buoc 1", "Giai nen va copy du lieu",
         ["Giai nen FraudShield_Project.zip",
          "Copy FraudShield_Banking_Data.csv vao thu muc FraudShield/data/"],
         C_ACCENT),
        ("Buoc 2", "Cai thu vien Python",
         ["Mo Terminal trong VS Code (Ctrl + `)",
          "Chay: pip install -r requirements.txt",
          "Cho den khi cai xong (khoang 1-2 phut)"],
         C_ACCENT),
        ("Buoc 3", "Sua config email (neu dung scheduler)",
         ["Mo file config.py",
          "Sua EMAIL_SENDER = 'your_gmail@gmail.com'",
          "Sua EMAIL_PASSWORD = '...app password...'",
          "Vao myaccount.google.com/apppasswords de lay App Password"],
         C_YELLOW),
        ("Buoc 4", "Chay Dashboard (cach don gian nhat)",
         ["Mo Terminal, chay: streamlit run app.py",
          "Trinh duyet tu mo tai http://localhost:8501",
          "Click 'Chay Pipeline Day du' o sidebar → cho model train xong",
          "Dung Tab 3 de upload CSV moi va du doan"],
         C_GREEN),
        ("Buoc 5", "Chay Scheduler (tu dong hoa hoan toan)",
         ["Mo Terminal moi (de it nhat 2 cua so terminal)",
          "Chay: python scheduler.py",
          "Tha bat ky file CSV nao vao thu muc inbox/",
          "He thong tu dong xu ly va gui email trong 60 giay"],
         colors.HexColor("#7c3aed")),
    ]

    for num, title, steps_list, color in installs:
        story.append(step_card(num, title, steps_list, color=color))
        story.append(Spacer(1, 0.15*cm))

    story.append(Spacer(1, 0.3*cm))
    story.append(info_box(
        "<b>Luu y quan trong:</b><br/>"
        "• Gmail App Password KHAC voi mat khau Gmail thuong. Phai tao rieng tai "
        "myaccount.google.com/apppasswords<br/>"
        "• Phai bat 2-Step Verification truoc khi tao App Password<br/>"
        "• Khong chia se App Password cho nguoi khac",
        bg=colors.HexColor("#fff7ed"), border=C_RED))

    story.append(Spacer(1, 0.5*cm))
    story.append(hr(C_BORDER, 1))
    story.append(Paragraph(
        "FraudShield — FAA4023 Final Assignment 2025–2026  |  "
        "Cong cu: Python, pandas, scikit-learn, Streamlit, matplotlib, openpyxl",
        ParagraphStyle("footer", fontName="Helvetica", fontSize=8.5,
            textColor=C_MUTED, alignment=TA_CENTER)))

    doc.build(story)
    print(f"[✓] PDF da tao: {OUT_PATH}")


if __name__ == "__main__":
    build()
