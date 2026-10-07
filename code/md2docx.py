# -*- coding: utf-8 -*-
"""
md2docx.py —— 将论文 Markdown 转换为规范 Word 文档

转换特性（对照课件格式检查清单）：
  - 封面页（题名、专业、指导教师等占位）
  - 自动目录页（TOC 域，Word 中按 F9 刷新页码）
  - 标题层级 1 / 1.1 / 1.1.1 映射 Heading 1/2/3
  - 图表编号题注：图注在图下、表注在表上（宋体加粗居中）
  - 表格自动套用三线表样式（Table Grid + 手工调线）
  - 参考文献悬挂缩进
  - 正文：宋体/思源宋体 小四（12pt），1.5 倍行距，首行缩进 2 字符
"""

import os
import re
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR

MD_PATH = os.path.join(BASE_DIR, "docs", "论文全文.md")
OUT_PATH = os.path.join(BASE_DIR, "论文全文.docx")

CJK = "思源宋体 CN"
CJK_FALLBACK = "宋体"
ASCII_FONT = "Times New Roman"


def set_cjk(run, size=12, bold=False, font=CJK):
    run.font.name = ASCII_FONT
    r = run._element.rPr
    rFonts = r.rFonts if r.rFonts is not None else r.get_or_add_rFonts()
    rFonts.set(qn("w:eastAsia"), font)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)


def para(doc, text="", size=12, bold=False, align=None, indent_chars=2,
         space_after=6, line=1.5, font=CJK):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    if indent_chars and text:
        pf.first_line_indent = Pt(size * indent_chars)
    run = p.add_run(text)
    set_cjk(run, size=size, bold=bold, font=font)
    return p


def add_toc(doc):
    """插入 TOC 域（打开 Word 后按 Ctrl+A → F9 刷新）"""
    p = doc.add_paragraph()
    run = p.add_run()
    fldChar = run._element.makeelement(qn("w:fldChar"), {qn("w:fldCharType"): "begin"})
    instrText = run._element.makeelement(
        qn("w:instrText"),
        {qn("xml:space"): "preserve"})
    instrText.text = r'TOC \o "1-3" \h \z \u'
    fldChar2 = run._element.makeelement(qn("w:fldChar"), {qn("w:fldCharType"): "separate"})
    t = run._element.makeelement(qn("w:t"), {})
    t.text = "右键更新域以生成目录"
    fldChar3 = run._element.makeelement(qn("w:fldChar"), {qn("w:fldCharType"): "end"})
    for el in (fldChar, instrText, fldChar2, t, fldChar3):
        run._element.append(el)


def style_table_three_line(table):
    """三线表：上下粗线 + 表头下细线"""
    tbl = table._tbl
    # 清除默认边框
    for row_idx, row in enumerate(table.rows):
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            borders = tcPr.makeelement(qn("w:tcBorders"), {})
            for side in ("top", "bottom", "left", "right"):
                el = tcPr.makeelement(qn(f"w:{side}"), {})
                if row_idx == 0 and side == "top":
                    el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "12")
                elif row_idx == 0 and side == "bottom":
                    el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "6")
                elif row_idx == len(table.rows) - 1 and side == "bottom":
                    el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "12")
                else:
                    el.set(qn("w:val"), "nil")
                borders.append(el)
            tcPr.append(borders)


def add_table_from_rows(doc, header, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        cell = table.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        set_cjk(run, size=9, bold=True)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            if j >= len(header):
                continue
            cell = table.rows[i].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(v)
            set_cjk(run, size=9)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_table_three_line(table)
    return table


def flush_table(doc, buf_rows):
    """把累积的 markdown 表格行写入 docx"""
    if not buf_rows:
        return
    header = [c.strip() for c in buf_rows[0].strip("|").split("|")]
    rows = []
    for line in buf_rows[2:]:
        rows.append([c.strip() for c in line.strip("|").split("|")])
    add_table_from_rows(doc, header, rows)
    buf_rows.clear()


def convert():
    with open(MD_PATH, encoding="utf-8") as f:
        lines = f.read().splitlines()

    doc = Document()
    # 页面设置：A4，2.5cm 边距
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.8)
    sec.top_margin = sec.bottom_margin = Cm(2.5)

    # ---- 封面 ----
    for _ in range(4):
        doc.add_paragraph()
    para(doc, "本科毕业论文（设计）", size=22, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0, space_after=30)
    para(doc, "企业ESG话语的文本测度构建与漂绿识别研究", size=18, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0, space_after=10)
    para(doc, "——基于2019—2023年A股上市公司ESG报告与年报MD&A的NLP实证分析",
         size=13, align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0, space_after=40)
    for label in ("学　　院：大数据与人工智能学院",
                  "专　　业：大数据管理与应用",
                  "学生姓名：×××（学号：××××××××）",
                  "指导教师：×××",
                  "完成日期：2026年×月"):
        para(doc, label, size=14, align=WD_ALIGN_PARAGRAPH.CENTER,
             indent_chars=0, space_after=10)
    doc.add_page_break()

    # ---- 目录页 ----
    para(doc, "目　录", size=16, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0, space_after=18)
    add_toc(doc)
    doc.add_page_break()

    # ---- 正文 ----
    in_code = False
    code_buf = []
    table_buf = []
    i = 0
    # 跳过 markdown 中的封面表格与标题（已用 docx 封面替代）
    skip_until_abstract = True

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            flush_table(doc, table_buf)
            if in_code:
                # 输出代码块
                for cl in code_buf:
                    p = doc.add_paragraph()
                    pf = p.paragraph_format
                    pf.space_after = Pt(0)
                    pf.line_spacing = 1.0
                    pf.left_indent = Cm(0.6)
                    run = p.add_run(cl)
                    run.font.name = "Consolas"
                    run.font.size = Pt(8.5)
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # 表格缓冲
        if line.strip().startswith("|"):
            table_buf.append(line)
            i += 1
            continue
        else:
            flush_table(doc, table_buf)

        s = line.strip()
        if not s:
            i += 1
            continue

        if skip_until_abstract:
            # 跳过 md 顶部的元信息标题与表格
            if s.startswith("## 摘") or s.startswith("## 摘要"):
                skip_until_abstract = False
            else:
                i += 1
                continue

        # 图片
        m = re.match(r"!\[(.*?)\]\((.*?)\)", s)
        if m:
            caption, rel = m.groups()
            img_path = os.path.join(BASE_DIR, "docs", rel)
            if not os.path.exists(img_path):
                img_path = os.path.join(BASE_DIR, rel)
            if os.path.exists(img_path):
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(img_path, width=Cm(14.5))
            i += 1
            continue

        # 图表题注（**图x-x ...** / **表x-x ...**）
        m = re.match(r"\*\*(图\d[\d\-]*[^\*]*|表\d[\d\-]*[^\*]*)\*\*", s)
        if m:
            cap = m.group(1).strip()
            para(doc, cap, size=10.5, bold=True,
                 align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
                 space_after=10, line=1.2)
            i += 1
            continue

        # 标题
        if s.startswith("### "):
            h = s[4:].strip()
            hp = doc.add_heading("", level=3)
            run = hp.add_run(h)
            set_cjk(run, size=13, bold=True)
            hp.paragraph_format.space_before = Pt(10)
            hp.paragraph_format.space_after = Pt(6)
            i += 1
            continue
        if s.startswith("## "):
            h = s[3:].strip()
            if h.startswith("——"):
                # 副标题并入正文处理
                para(doc, h, size=13, align=WD_ALIGN_PARAGRAPH.CENTER,
                     indent_chars=0, space_after=14)
                i += 1
                continue
            hp = doc.add_heading("", level=1)
            run = hp.add_run(h)
            set_cjk(run, size=15, bold=True)
            hp.paragraph_format.space_before = Pt(14)
            hp.paragraph_format.space_after = Pt(8)
            i += 1
            continue
        if s.startswith("# "):
            i += 1
            continue

        # 摘要关键词等特殊行
        if s.startswith("**关键词**"):
            para(doc, s.replace("**", ""), size=12, indent_chars=0,
                 space_after=10)
            i += 1
            continue

        # 分隔线
        if s == "---":
            i += 1
            continue

        # 参考文献条目（[1] 开头）：悬挂缩进
        if re.match(r"^\[\d+\]", s):
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.left_indent = Cm(0.85)
            pf.first_line_indent = Cm(-0.85)
            pf.space_after = Pt(4)
            pf.line_spacing = 1.3
            run = p.add_run(s)
            set_cjk(run, size=10.5)
            i += 1
            continue

        # > 引用块
        if s.startswith("> "):
            p = para(doc, s[2:], size=10.5, indent_chars=0, space_after=8)
            p.paragraph_format.left_indent = Cm(0.8)
            p.paragraph_format.right_indent = Cm(0.8)
            i += 1
            continue

        # 普通正文段（去除行内 markdown 标记）
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"`(.+?)`", r"\1", text)
        text = re.sub(r"\[(\d+)\]\((http.*?)\)", r"[\1]", text)
        para(doc, text, size=12, indent_chars=2, space_after=6)
        i += 1

    flush_table(doc, table_buf)
    doc.save(OUT_PATH)
    print(f"Word 文档已生成: {OUT_PATH}")


if __name__ == "__main__":
    convert()
