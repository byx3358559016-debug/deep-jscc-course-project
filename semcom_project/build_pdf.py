"""Build the final PDF directly from 技术报告.md using ReportLab."""

from __future__ import annotations

import html
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = PROJECT_DIR / "技术报告.md"
OUTPUT_PATH = PROJECT_DIR.parent / "output" / "pdf" / "语义通信课程设计技术报告.pdf"

NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2E74B5")
DARK_BLUE = colors.HexColor("#1F4D78")
LIGHT_BLUE = colors.HexColor("#DCE6F1")
LIGHT_GRAY = colors.HexColor("#F2F4F7")
MUTED = colors.HexColor("#666666")
GOLD = colors.HexColor("#B08D32")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("SimSun", r"C:\Windows\Fonts\simsun.ttc", subfontIndex=0))
    pdfmetrics.registerFont(TTFont("SimHei", r"C:\Windows\Fonts\simhei.ttf"))
    pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
    pdfmetrics.registerFont(TTFont("ArialBold", r"C:\Windows\Fonts\arialbd.ttf"))


class AcademicReportTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs) -> None:
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=self._header_footer)])

    def _header_footer(self, canvas, doc) -> None:
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setFont("SimSun", 8.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(self.leftMargin, A4[1] - 14 * mm, "现代通信技术实践  |  Deep JSCC 图像语义通信")
        canvas.setStrokeColor(colors.HexColor("#D7DEE8"))
        canvas.setLineWidth(0.4)
        canvas.line(self.leftMargin, A4[1] - 15.5 * mm, A4[0] - self.rightMargin, A4[1] - 15.5 * mm)
        canvas.drawCentredString(A4[0] / 2, 13 * mm, f"第 {doc.page - 1} 页")
        canvas.restoreState()

    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name in {"ReportH1", "ReportH2"}:
            level = 0 if flowable.style.name == "ReportH1" else 1
            text = flowable.getPlainText()
            key = f"heading-{level}-{self.seq.nextf('heading')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=False)
            self.notify("TOCEntry", (level, text, self.page - 1, key))


def make_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "ReportBody",
            parent=styles["BodyText"],
            fontName="SimSun",
            fontSize=10.5,
            leading=18,
            alignment=TA_JUSTIFY,
            firstLineIndent=21,
            spaceAfter=6,
            wordWrap="CJK",
            textColor=colors.black,
        ),
        "h1": ParagraphStyle(
            "ReportH1",
            parent=styles["Heading1"],
            fontName="SimSun",
            fontSize=16,
            leading=23,
            textColor=NAVY,
            spaceBefore=14,
            spaceAfter=8,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "ReportH2",
            parent=styles["Heading2"],
            fontName="SimSun",
            fontSize=13,
            leading=19,
            textColor=BLUE,
            spaceBefore=11,
            spaceAfter=6,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "ReportCaption",
            parent=styles["BodyText"],
            fontName="SimSun",
            fontSize=9,
            leading=13,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "equation": ParagraphStyle(
            "ReportEquation",
            parent=styles["BodyText"],
            fontName="SimSun",
            fontSize=10.5,
            leading=17,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=7,
            wordWrap="CJK",
        ),
        "reference": ParagraphStyle(
            "ReportReference",
            parent=styles["BodyText"],
            fontName="SimSun",
            fontSize=9.2,
            leading=14,
            leftIndent=18,
            firstLineIndent=-18,
            alignment=TA_LEFT,
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "ReportCode",
            parent=styles["Code"],
            fontName="SimSun",
            fontSize=8.5,
            leading=12,
            leftIndent=12,
            rightIndent=8,
            backColor=LIGHT_GRAY,
            borderPadding=6,
            borderColor=colors.HexColor("#D9DEE5"),
            borderWidth=0.4,
            spaceBefore=4,
            spaceAfter=6,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            fontName="SimSun",
            fontSize=25,
            leading=35,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=15,
            wordWrap="CJK",
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            fontName="SimSun",
            fontSize=16,
            leading=23,
            textColor=BLUE,
            alignment=TA_CENTER,
            spaceAfter=40,
            wordWrap="CJK",
        ),
        "cover_kicker": ParagraphStyle(
            "CoverKicker",
            fontName="SimSun",
            fontSize=12,
            leading=18,
            textColor=GOLD,
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "toc_title": ParagraphStyle(
            "TOCTitle",
            fontName="SimSun",
            fontSize=16,
            leading=22,
            alignment=TA_CENTER,
            textColor=NAVY,
            spaceAfter=12,
        ),
    }


def clean(text: str) -> str:
    return text.replace("`", "").replace("**", "")


def safe(text: str) -> str:
    return html.escape(clean(text)).replace("\n", "<br/>")


def cover_story(styles) -> list:
    label_style = ParagraphStyle("CoverLabel", fontName="SimHei", fontSize=10.5, textColor=NAVY, leading=16)
    value_style = ParagraphStyle("CoverValue", fontName="SimSun", fontSize=10.5, textColor=colors.black, leading=16)
    data = [
        [Paragraph("课程", label_style), Paragraph("现代通信技术实践", value_style)],
        [Paragraph("方向", label_style), Paragraph("基于深度学习的语义通信系统设计与实践", value_style)],
        [Paragraph("小组成员 / 学号", label_style), Paragraph("____________________________", value_style)],
        [Paragraph("指导教师", label_style), Paragraph("____________________________", value_style)],
        [Paragraph("完成日期", label_style), Paragraph("2026 年 8 月", value_style)],
    ]
    table = Table(data, colWidths=[40 * mm, 95 * mm], hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.55, colors.HexColor("#9FB3C8")),
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return [
        Spacer(1, 28 * mm),
        Paragraph("现代通信技术实践 · 课程设计技术报告", styles["cover_kicker"]),
        Paragraph("基于深度学习的图像语义通信系统<br/>设计与实践", styles["cover_title"]),
        Paragraph("Deep JSCC 复现、模型改进与鲁棒性分析", styles["cover_subtitle"]),
        table,
        PageBreak(),
    ]


def table_flowable(rows: list[list[str]], styles) -> LongTable:
    column_count = len(rows[0])
    weights = []
    for column in range(column_count):
        longest = max(len(row[column]) for row in rows)
        weights.append(max(5.0, min(24.0, float(longest))))
    if column_count >= 4:
        weights[0] = max(weights[0], 13.0)
    usable = 159 * mm
    total = sum(weights)
    widths = [usable * weight / total for weight in weights]
    size = 7.7 if column_count >= 7 else 8.1 if column_count >= 6 else 8.6 if column_count >= 5 else 9.0
    header_style = ParagraphStyle("TableHeader", fontName="SimHei", fontSize=size, leading=size + 3, alignment=TA_CENTER, wordWrap="CJK")
    cell_style = ParagraphStyle("TableCell", fontName="SimSun", fontSize=size, leading=size + 3, alignment=TA_CENTER, wordWrap="CJK")
    first_style = ParagraphStyle("TableFirst", parent=cell_style, alignment=TA_LEFT)
    data = []
    for row_index, row in enumerate(rows):
        data.append(
            [
                Paragraph(safe(value), header_style if row_index == 0 else first_style if column_index == 0 else cell_style)
                for column_index, value in enumerate(row)
            ]
        )
    table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAB7C4")),
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BLUE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def image_flowables(path: Path, caption: str, styles) -> list:
    with PILImage.open(path) as source:
        width, height = source.size
    max_width = 158 * mm
    max_height = 92 * mm
    scale = min(max_width / width, max_height / height)
    image = Image(str(path), width=width * scale, height=height * scale)
    image.hAlign = "CENTER"
    return [KeepTogether([image, Paragraph(safe(caption), styles["caption"])])]


def parse_markdown(lines: list[str], styles) -> list:
    story = cover_story(styles)
    index = 0
    in_code = False
    code_lines: list[str] = []
    toc_added = False
    while index < len(lines):
        raw = lines[index].rstrip()
        line = raw.strip()
        if line.startswith("# ") or line.startswith(("课程：", "方向：", "小组成员：", "指导教师：", "完成日期：")):
            index += 1
            continue
        if line.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["code"])); code_lines = []; in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code_lines.append(raw); index += 1; continue
        if not line:
            index += 1; continue
        if line.startswith("## "):
            text = line[3:].strip()
            if text == "1 引言" and not toc_added:
                toc = TableOfContents()
                toc.levelStyles = [
                    ParagraphStyle("TOC1", fontName="SimSun", fontSize=10.5, leading=18, leftIndent=0, firstLineIndent=0, spaceBefore=2),
                    ParagraphStyle("TOC2", fontName="SimSun", fontSize=9.5, leading=15, leftIndent=16, firstLineIndent=0, spaceBefore=1, textColor=MUTED),
                ]
                story.extend([PageBreak(), Paragraph("目录", styles["toc_title"]), toc, PageBreak()])
                toc_added = True
            if text in {"附录 A 代码结构与复现命令", "附录 B 课程要求覆盖对照"}:
                story.append(PageBreak())
            story.append(Paragraph(safe(text), styles["h1"]))
            index += 1; continue
        if line.startswith("### "):
            subsection = line[4:].strip()
            story.append(Paragraph(safe(subsection), styles["h2"]))
            index += 1; continue
        image_match = re.match(r"!\[(.+?)\]\((.+?)\)", line)
        if image_match:
            caption, relative_path = image_match.groups()
            story.extend(image_flowables(PROJECT_DIR / relative_path, caption, styles))
            index += 1; continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip()); index += 1
            rows = []
            for table_line in table_lines:
                cells = [cell.strip() for cell in table_line.strip("|").split("|")]
                if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    continue
                rows.append(cells)
            story.extend([table_flowable(rows, styles), Spacer(1, 5)])
            continue
        paragraph_lines = [line]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate or candidate.startswith(("#", "|", "```", "![")):
                break
            paragraph_lines.append(candidate); index += 1
        text = clean(" ".join(paragraph_lines))
        if text.startswith("关键词："):
            label, value = text.split("：", 1)
            story.append(Paragraph(f"<b>{safe(label)}：</b>{safe(value)}", ParagraphStyle("Keywords", parent=styles["body"], firstLineIndent=0, spaceAfter=10)))
        elif len(text) < 90 and any(token in text for token in ("arg min", "log10", "CN(", "k/n =", "E[z", "z =", "y =")):
            story.append(Paragraph(safe(text), styles["equation"]))
        elif re.match(r"^\[\d+\]", text):
            story.append(Paragraph(safe(text), styles["reference"]))
        else:
            story.append(Paragraph(safe(text), styles["body"]))
    return story


def main() -> None:
    register_fonts()
    styles = make_styles()
    story = parse_markdown(SOURCE_PATH.read_text(encoding="utf-8").splitlines(), styles)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = AcademicReportTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=25.4 * mm,
        rightMargin=25.4 * mm,
        topMargin=22 * mm,
        bottomMargin=22 * mm,
        title="基于深度学习的图像语义通信系统设计与实践",
        author="课程设计小组",
        subject="现代通信技术实践课程设计",
    )
    document.multiBuild(story)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
