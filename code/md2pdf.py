# -*- coding: utf-8 -*-
"""
md2pdf.py —— 论文 Markdown → HTML → PDF（weasyprint 排版）
"""

import os
import re
import subprocess
import sys

import markdown

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR

MD_PATH = os.path.join(BASE_DIR, "docs", "论文全文.md")
HTML_PATH = os.path.join(BASE_DIR, "论文全文.html")
PDF_PATH = os.path.join(BASE_DIR, "论文全文.pdf")

CSS = """
@page {
    size: A4;
    margin: 2.5cm 2.6cm;
    @bottom-center { content: counter(page); font-size: 9pt; color: #666; }
}
@page :first { @bottom-center { content: none; } }
body {
    font-family: "Noto Serif CJK SC", "Noto Sans CJK SC", serif;
    font-size: 11.5pt; line-height: 1.75; color: #1a1a1a;
}
h1 { font-size: 19pt; text-align: center; margin: 0.4em 0; }
h1 + h2 { font-size: 12.5pt; text-align: center; font-weight: normal;
          color: #444; border: none; }
h2 { font-size: 15pt; margin: 1.2em 0 0.6em; border-bottom: 2px solid #2E5A88;
     padding-bottom: 4px; page-break-after: avoid; }
h3 { font-size: 12.5pt; margin: 1em 0 0.4em; page-break-after: avoid; }
p { text-align: justify; text-indent: 2em; margin: 0.45em 0; }
h1, h2, h3, h4, table, figure, img, .caption, .noindent { }
h1 + h2 + p, .meta p, .caption, figcaption, .kw, li, td, th, pre, .refs p
    { text-indent: 0; }
table { border-collapse: collapse; margin: 0.7em auto; width: 96%;
        font-size: 9pt; page-break-inside: avoid; }
thead { border-top: 1.6pt solid #333; border-bottom: 0.8pt solid #333; }
tbody tr:last-child { border-bottom: 1.6pt solid #333; }
th, td { padding: 3.5pt 6pt; text-align: center; }
th { font-weight: 700; }
tr { border-bottom: 0.4pt solid #ccc; }
img { max-width: 88%; display: block; margin: 0.6em auto; }
.caption, p strong:only-child { display: block; text-align: center;
    font-size: 9.5pt; font-weight: 700; margin: 0.2em 0 0.9em;
    text-indent: 0; }
pre { background: #f6f8fa; border: 0.6pt solid #ddd; border-radius: 4px;
      padding: 8pt 10pt; font-family: "Noto Sans Mono CJK SC", monospace;
      font-size: 8pt; line-height: 1.45; page-break-inside: avoid;
      white-space: pre-wrap; }
code { font-family: "Noto Sans Mono CJK SC", monospace; font-size: 8.5pt;
       background: #f2f4f7; padding: 0 3px; border-radius: 3px; }
pre code { background: none; padding: 0; }
blockquote { margin: 0.8em 1.2em; padding: 6pt 12pt; background: #f0f4f9;
             border-left: 3pt solid #2E5A88; font-size: 10pt; color: #333; }
blockquote p { text-indent: 0; margin: 0.3em 0; }
.refs p { text-indent: -1.7em; margin-left: 1.7em; font-size: 10pt;
          margin-top: 0.25em; }
.kw p { text-indent: 0; }
hr { border: none; border-top: 0.7pt solid #bbb; margin: 1.4em 0; }
.meta table { font-size: 10pt; }
"""


def md_to_html():
    with open(MD_PATH, encoding="utf-8") as f:
        text = f.read()

    md = markdown.Markdown(extensions=["tables", "fenced_code"])
    body = md.convert(text)

    # 参考文献段落打标（悬挂缩进）
    body = re.sub(r"<p>(\[\d+\])", r'<p class="ref">\1', body)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>企业ESG话语的文本测度构建与漂绿识别研究</title>
<style>{CSS}</style></head><body>{body}</body></html>"""
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML 已生成: {HTML_PATH}")


def html_to_pdf():
    from weasyprint import HTML
    HTML(HTML_PATH, base_url=BASE_DIR).write_pdf(PDF_PATH)
    print(f"PDF 已生成: {PDF_PATH}")


if __name__ == "__main__":
    md_to_html()
    html_to_pdf()
