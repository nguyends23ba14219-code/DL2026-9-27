from pathlib import Path
import html
import re
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether

root = Path(__file__).resolve().parents[1]
source = root / "reports/report_vi.md"
font_pairs = [
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]
font_pair = next((pair for pair in font_pairs if all(Path(file).exists() for file in pair)), None)
if font_pair is None:
    raise RuntimeError("Install Arial or DejaVu Sans with Vietnamese glyphs to export the report")
pdfmetrics.registerFont(TTFont("Arial", font_pair[0]))
pdfmetrics.registerFont(TTFont("ArialBold", font_pair[1]))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="ArialBold", italic="Arial", boldItalic="ArialBold")
styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        "BodyVi",
        fontName="Arial",
        fontSize=10.5,
        leading=15,
        spaceAfter=9,
        splitLongWords=True,
        textColor=colors.HexColor("#26374d"),
    )
)
styles.add(
    ParagraphStyle(
        "TitleVi",
        fontName="ArialBold",
        fontSize=23,
        leading=28,
        spaceAfter=22,
        keepWithNext=True,
        textColor=colors.HexColor("#26374d"),
    )
)
styles.add(
    ParagraphStyle(
        "HeadingVi",
        fontName="ArialBold",
        fontSize=15,
        leading=20,
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True,
        textColor=colors.HexColor("#007f73"),
    )
)
styles.add(ParagraphStyle("CellVi", fontName="Arial", fontSize=9, leading=13))
styles.add(ParagraphStyle("HeaderVi", fontName="ArialBold", fontSize=9, leading=13, textColor=colors.white))
styles.add(
    ParagraphStyle(
        "CaptionVi", fontName="Arial", fontSize=9, leading=12, textColor=colors.HexColor("#536070"), spaceAfter=12
    )
)


def inline(text):
    text = html.escape(text.replace("—", "-").replace("–", "-"))
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font color="#007f73">\1</font>', text)
    return text


lines = source.read_text().splitlines()
flow = []
i = 0
while i < len(lines):
    line = lines[i].strip()
    if not line:
        i += 1
        continue
    if line.startswith("# "):
        flow.append(Paragraph(inline(line[2:]), styles["TitleVi"]))
    elif line.startswith("## "):
        flow.append(Paragraph(inline(line[3:]), styles["HeadingVi"]))
    elif line.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            row = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
            if not all(re.match(r"^[-:]+$", cell) for cell in row):
                rows.append(row)
            i += 1
        n = len(rows[0])
        widths = [499 / n] * n
        if n == 5:
            widths = [139, 122, 88, 95, 55]
        if n == 6:
            widths = [139, 62, 58, 82, 91, 67]
        if n == 4:
            widths = [160, 100, 119, 120]
        cells = [
            [Paragraph(inline(cell), styles["HeaderVi" if r == 0 else "CellVi"]) for cell in row]
            for r, row in enumerate(rows)
        ]
        table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#26374d")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f5f5"), colors.white]),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#d0dada")),
                ]
            )
        )
        flow.extend([table, Spacer(1, 12)])
        continue
    elif line.startswith("!["):
        match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
        image_path = (source.parent / match.group(2)).resolve()
        image = Image(str(image_path))
        factor = min(499 / image.imageWidth, 420 / image.imageHeight)
        image.drawWidth = image.imageWidth * factor
        image.drawHeight = image.imageHeight * factor
        flow.append(
            KeepTogether([Spacer(1, 6), image, Spacer(1, 5), Paragraph(inline(match.group(1)), styles["CaptionVi"])])
        )
    else:
        paragraph = [line]
        while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].strip().startswith(("#", "|", "![")):
            i += 1
            paragraph.append(lines[i].strip())
        flow.append(Paragraph(inline(" ".join(paragraph)), styles["BodyVi"]))
    i += 1


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 8)
    canvas.setFillColor(colors.HexColor("#536070"))
    canvas.drawString(48, 25, "Project 27 - Fashion-MNIST / FedAvg / Non-IID")
    canvas.drawRightString(A4[0] - 48, 25, str(doc.page))
    canvas.restoreState()


pdf = root / "reports/report_vi.pdf"
SimpleDocTemplate(
    str(pdf),
    pagesize=A4,
    rightMargin=48,
    leftMargin=48,
    topMargin=42,
    bottomMargin=45,
    title="Project 27 - Federated Image Classification under Non-IID Data",
    author="nguyends23ba14219-code",
).build(flow, onFirstPage=decorate, onLaterPages=decorate)
print(pdf)
