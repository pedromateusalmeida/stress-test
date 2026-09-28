r"""Renderiza o relatório Markdown em PDF A4 com figuras e tabelas.

Uso: `python scripts/renderizar_relatorio_pdf.py`, após instalar requirements.txt.
O Markdown é a fonte editável; o PDF contém a mesma ordem de seções e imagens.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
import reportlab

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, LongTable, PageTemplate,
    Paragraph, Spacer, TableStyle,
)

BASE = Path(__file__).resolve().parents[1] / "reports"
SOURCE = BASE / "RELATORIO_DIDATICO.md"
DEST = BASE / "RELATORIO_DIDATICO.pdf"
_candidatas = [
    (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
    (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
     Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    (Path(reportlab.__file__).resolve().parent / "fonts" / "Vera.ttf",
     Path(reportlab.__file__).resolve().parent / "fonts" / "VeraBd.ttf"),
]
FONT, FONT_BOLD = next(
    ((normal, bold) for normal, bold in _candidatas
     if normal.is_file() and bold.is_file()),
    (None, None),
)
if FONT is None:
    raise FileNotFoundError("Não encontrei fontes TrueType para o PDF.")
pdfmetrics.registerFont(TTFont("ArialDoc", str(FONT)))
pdfmetrics.registerFont(TTFont("ArialDoc-Bold", str(FONT_BOLD)))
pdfmetrics.registerFontFamily("ArialDoc", normal="ArialDoc", bold="ArialDoc-Bold")

NAVY = colors.HexColor("#173451")
TEAL = colors.HexColor("#167D91")
PALE = colors.HexColor("#EDF5F7")
TEXT = colors.HexColor("#253244")
GREY = colors.HexColor("#5B6675")
PAGE_W, PAGE_H = A4
LEFT = RIGHT = 1.75 * cm
TOP = 1.75 * cm
BOTTOM = 1.65 * cm
WIDTH = PAGE_W - LEFT - RIGHT

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ReportTitle", fontName="ArialDoc-Bold", fontSize=21, leading=25,
    textColor=NAVY, spaceAfter=11, alignment=TA_LEFT, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="Section", fontName="ArialDoc-Bold", fontSize=14.6, leading=18,
    textColor=NAVY, spaceBefore=13, spaceAfter=6, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="Subsection", fontName="ArialDoc-Bold", fontSize=11.5, leading=15,
    textColor=TEAL, spaceBefore=10, spaceAfter=4, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="Subsubsection", fontName="ArialDoc-Bold", fontSize=9.8, leading=13,
    textColor=NAVY, spaceBefore=8, spaceAfter=4, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="ReportBody", fontName="ArialDoc", fontSize=8.95, leading=13.2,
    textColor=TEXT, alignment=TA_LEFT, spaceAfter=5.5,
    allowWidows=0, allowOrphans=0,
))
styles.add(ParagraphStyle(
    name="ReportBullet", parent=styles["ReportBody"], leftIndent=13,
    firstLineIndent=-11, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="ReportQuote", parent=styles["ReportBody"], leftIndent=13,
    rightIndent=11, textColor=GREY,
))
styles.add(ParagraphStyle(
    name="ReportCaption", fontName="ArialDoc", fontSize=7.8, leading=10.2,
    textColor=GREY, alignment=TA_LEFT, spaceAfter=9,
))
styles.add(ParagraphStyle(
    name="ReportTable", fontName="ArialDoc", fontSize=6.75, leading=9.1,
    textColor=TEXT, spaceAfter=0, splitLongWords=1,
))
styles.add(ParagraphStyle(
    name="ReportTableHeader", parent=styles["ReportTable"],
    fontName="ArialDoc-Bold", textColor=colors.white,
))


def inline_md(value: str, in_table: bool = False) -> str:
    """Subset seguro de Markdown inline aceito por Paragraph do ReportLab."""
    value = html.escape(value, quote=False)
    value = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        lambda m: f'<link href="{html.escape(m.group(2), quote=True)}" color="#126A85">{m.group(1)}</link>',
        value,
    )
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    if in_table:
        value = re.sub(r"`([^`]+)`", r"\1", value)
    else:
        value = re.sub(r"`([^`]+)`", r'<font name="ArialDoc" size="8">\1</font>', value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    return value.replace("  ", " ")


def paragraph(text: str, style="ReportBody") -> Paragraph:
    return Paragraph(inline_md(text, in_table=style.startswith("ReportTable")), styles[style])


def table_widths(header: list[str], n: int) -> list[float]:
    if n == 2:
        weights = [0.38, 0.62]
    elif n == 3:
        weights = [0.40, 0.29, 0.31]
    elif n == 4:
        weights = [0.25, 0.17, 0.27, 0.31]
    elif n == 5:
        weights = [0.36, 0.21, 0.15, 0.15, 0.13]
    else:
        weights = [0.38] + [0.62 / (n - 1)] * (n - 1)
    if "Fonte recebida" in header:
        weights = [0.23, 0.14, 0.30, 0.33]
    if "Variável" in header and n == 5:
        weights = [0.33, 0.21, 0.16, 0.15, 0.15]
    return [WIDTH * w for w in weights]


def parse_table(lines: list[str]) -> LongTable:
    raw = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    raw = [row for row in raw if not all(re.fullmatch(r":?-{2,}:?", c) for c in row)]
    n = max(map(len, raw))
    data = []
    for i, row in enumerate(raw):
        row = row + [""] * (n - len(row))
        style = "ReportTableHeader" if i == 0 else "ReportTable"
        data.append([paragraph(c, style) for c in row])
    widths = table_widths(raw[0], n)
    table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, 0), .8, TEAL),
        ("LINEBELOW", (0, 1), (-1, -1), .25, colors.HexColor("#D7E3E7")),
    ]))
    return table


def make_image(line: str) -> tuple[Image, str]:
    match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line.strip())
    if match is None:
        raise ValueError(line)
    alt, rel = match.groups()
    path = BASE / rel
    if not path.is_file():
        raise FileNotFoundError(path)
    img = Image(str(path))
    scale = min(WIDTH / img.imageWidth, 18.4 * cm / img.imageHeight, 1.0)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    img.hAlign = "CENTER"
    return img, alt


def parse_document(source: str) -> list:
    lines = source.splitlines()
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("!["):
            img, alt = make_image(line)
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            caption = None
            if i < len(lines) and lines[i].strip().startswith("*Legenda"):
                caption = paragraph(lines[i].strip(), "ReportCaption")
                i += 1
            else:
                caption = paragraph(alt, "ReportCaption")
            story.append(KeepTogether([img, Spacer(1, 4), caption]))
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            story.extend([Spacer(1, 3), parse_table(rows), Spacer(1, 8)])
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            levels = {1: "ReportTitle", 2: "Section", 3: "Subsection", 4: "Subsubsection"}
            story.append(paragraph(heading.group(2), levels[len(heading.group(1))]))
            i += 1
            continue
        if re.match(r"^(?:[-*]\s+|\d+\.\s+)", line):
            mark = re.match(r"^([-*]|\d+\.)\s+(.+)$", line)
            prefix = "•" if mark.group(1) in ("-", "*") else mark.group(1)
            story.append(paragraph(prefix + "  " + mark.group(2), "ReportBullet"))
            i += 1
            continue
        if line.startswith(">"):
            story.append(paragraph(line.lstrip("> "), "ReportQuote"))
            i += 1
            continue
        if line == "---":
            story.append(Spacer(1, 8))
            i += 1
            continue
        body = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or re.match(r"^(?:#{1,4}\s|[-*]\s|\d+\.\s|\||!\[|>)", nxt):
                break
            body.append(nxt)
            i += 1
        story.append(paragraph(" ".join(body)))
    return story


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#B8CCD5"))
    canvas.setLineWidth(.55)
    canvas.line(LEFT, PAGE_H - 1.27 * cm, PAGE_W - RIGHT, PAGE_H - 1.27 * cm)
    canvas.setFont("ArialDoc", 7.1)
    canvas.setFillColor(GREY)
    canvas.drawString(LEFT, PAGE_H - 1.05 * cm, "TESTES DE ESTRESSE E DRIFT · CRÉDITO")
    canvas.line(LEFT, 1.15 * cm, PAGE_W - RIGHT, 1.15 * cm)
    canvas.drawString(LEFT, .84 * cm, "Modelo XGBoost · peso 13,882494684620836 · corte 0,50")
    canvas.drawRightString(PAGE_W - RIGHT, .84 * cm, f"Página {doc.page}")
    canvas.restoreState()


class ReportDoc(BaseDocTemplate):
    pass


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    doc = ReportDoc(
        str(DEST), pagesize=A4, leftMargin=LEFT, rightMargin=RIGHT,
        topMargin=TOP, bottomMargin=BOTTOM,
        title="Testes de estresse e tolerância a drift do modelo de crédito",
        author="Codex / estudo reproduzível em Python",
    )
    frame = Frame(LEFT, BOTTOM, WIDTH, PAGE_H - TOP - BOTTOM,
                  leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0)
    doc.addPageTemplates(PageTemplate(id="Normal", frames=frame, onPage=header_footer))
    story = parse_document(SOURCE.read_text(encoding="utf-8"))
    doc.build(story)
    print(DEST)


if __name__ == "__main__":
    main()
